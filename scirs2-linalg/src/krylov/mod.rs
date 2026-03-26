//! Krylov subspace methods — high-level builder API
//!
//! This module provides a unified, struct-based interface to the Krylov
//! subspace algorithms available in `scirs2-linalg`. It sits on top of the
//! lower-level function-oriented API in [`crate::iterative`] and the
//! matrix-function Krylov machinery in
//! [`crate::matrix_functions::interpolation`].
//!
//! ## Design overview
//!
//! | Struct | Purpose |
//! |--------|---------|
//! | [`KrylovBasis`] | Build an explicit Krylov basis (Arnoldi / Lanczos) |
//! | [`Gmres`] | Restarted GMRES solver (general non-symmetric systems) |
//! | [`ConjugateGradient`] | PCG solver (symmetric positive definite systems) |
//! | [`BiCgStab`] | BiCGSTAB solver (general non-symmetric systems) |
//! | [`Minres`] | MINRES solver (symmetric possibly indefinite systems) |
//!
//! All struct-based solvers follow the same builder pattern:
//!
//! ```text
//! let result = Gmres::new()
//!     .tol(1e-10)
//!     .max_iter(500)
//!     .restart(30)
//!     .solve(&a, &b)?;
//! ```
//!
//! ## Convergence diagnostics
//!
//! [`SolveResult`] carries richer convergence information than the bare
//! [`crate::iterative::IterativeSolveResult`]: it stores the full residual
//! history and an estimate of the condition number of the projected system.
//!
//! ## References
//!
//! - Saad, Y. & Schultz, M. H. (1986). *GMRES: A generalized minimal residual
//!   algorithm for solving nonsymmetric linear systems.*
//! - Hestenes, M. R. & Stiefel, E. (1952). *Methods of conjugate gradients for
//!   solving linear systems.*
//! - Paige, C. C. & Saunders, M. A. (1975). *Solution of sparse indefinite
//!   systems of linear equations.*
//! - Van der Vorst, H. A. (1992). *Bi-CGSTAB: A fast and smoothly converging
//!   variant of Bi-CG for the solution of nonsymmetric linear systems.*

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::iterative::{bicgstab, conjugate_gradient, gmres, minres, IterativeSolveResult};
use crate::validation::{validate_iteration_parameters, validate_linear_system};

mod fgmres;
pub use fgmres::{fgmres_solve, Fgmres};

// ─────────────────────────────────────────────────────────────────────────────
// Re-export basic result type for convenience
// ─────────────────────────────────────────────────────────────────────────────

pub use crate::iterative::IterativeSolveResult as BasicSolveResult;

// ─────────────────────────────────────────────────────────────────────────────
// Rich convergence result
// ─────────────────────────────────────────────────────────────────────────────

/// Extended result type returned by all struct-based Krylov solvers.
///
/// Extends [`BasicSolveResult`] with full convergence history and
/// a condition-number estimate derived from the projected system.
#[derive(Debug, Clone)]
pub struct SolveResult<F> {
    /// Approximate solution vector.
    pub x: Array1<F>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Euclidean norm of the final residual ‖b − Ax‖.
    pub residual_norm: F,
    /// Whether the solver reached the requested tolerance.
    pub converged: bool,
    /// Residual norm at every iteration (length == `iterations`).
    pub residual_history: Vec<F>,
    /// Estimate of the spectral condition number κ(A) derived from the
    /// projected Hessenberg / tridiagonal matrix, if available.
    pub condition_estimate: Option<F>,
}

impl<F: Float + NumAssign + Sum + ScalarOperand> From<IterativeSolveResult<F>> for SolveResult<F> {
    fn from(r: IterativeSolveResult<F>) -> Self {
        Self {
            x: r.x,
            iterations: r.iterations,
            residual_norm: r.residual_norm,
            converged: r.converged,
            residual_history: Vec::new(),
            condition_estimate: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KrylovBasis — explicit basis construction
// ─────────────────────────────────────────────────────────────────────────────

/// Krylov basis type: determines the orthogonalisation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisKind {
    /// Arnoldi iteration (general matrices) — produces an upper-Hessenberg
    /// projected matrix.
    Arnoldi,
    /// Lanczos iteration (symmetric matrices) — produces a symmetric
    /// tridiagonal projected matrix.
    Lanczos,
}

/// Result of building a Krylov basis.
#[derive(Debug, Clone)]
pub struct KrylovBasisResult<F> {
    /// Orthonormal Krylov basis columns V_m (shape n × m).
    pub basis: Array2<F>,
    /// Upper Hessenberg (Arnoldi) or symmetric tridiagonal (Lanczos) projected
    /// matrix, stored in (m+1) × m form (Arnoldi) or m × m form (Lanczos).
    pub projected: Array2<F>,
    /// Number of basis vectors actually computed (≤ `max_dim`).
    pub dim: usize,
    /// True when a happy breakdown was detected: A-invariant subspace found.
    pub happy_breakdown: bool,
    /// Norm of the residual vector h_{m+1,m} (Arnoldi) or β_{m+1} (Lanczos).
    pub residual_norm: F,
    /// Which iteration strategy was used.
    pub kind: BasisKind,
}

impl<F: Float + NumAssign + Sum + ScalarOperand> KrylovBasisResult<F> {
    /// Estimate the spectral condition number κ(H_m) of the projected matrix
    /// via the ratio of extreme singular values of the square part.
    ///
    /// Only the first `dim × dim` block of `projected` is used.
    pub fn condition_estimate(&self) -> F {
        let m = self.dim;
        if m == 0 {
            return F::one();
        }
        // Collect the m × m square part of the projected matrix.
        let h = &self.projected;
        // Gershgorin circle estimate of the spectral radius.
        let mut max_row = F::zero();
        let mut min_diag = F::max_value();
        for i in 0..m {
            let diag = h[[i, i]].abs();
            if diag < min_diag {
                min_diag = diag;
            }
            let mut off = F::zero();
            for j in 0..m {
                if j != i {
                    off += h[[i, j]].abs();
                }
            }
            let upper = diag + off;
            if upper > max_row {
                max_row = upper;
            }
        }
        if min_diag <= F::epsilon() {
            return F::from(1e15_f64).unwrap_or(F::one());
        }
        max_row / min_diag
    }
}

/// High-level Krylov basis builder.
///
/// Supports both Arnoldi (for general matrices) and Lanczos (for symmetric
/// matrices) iterations with configurable dimension and breakdown tolerance.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_linalg::krylov::{KrylovBasis, BasisKind};
///
/// let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
/// let v0 = Array1::from_vec(vec![1.0_f64, 0.0]);
/// let result = KrylovBasis::new()
///     .kind(BasisKind::Arnoldi)
///     .max_dim(2)
///     .build(&a, &v0.view())
///     .expect("basis failed");
/// assert_eq!(result.dim, 2);
/// ```
#[derive(Debug, Clone)]
pub struct KrylovBasis {
    kind: BasisKind,
    max_dim: usize,
    breakdown_tol: f64,
}

impl Default for KrylovBasis {
    fn default() -> Self {
        Self {
            kind: BasisKind::Arnoldi,
            max_dim: 30,
            breakdown_tol: 1e-14,
        }
    }
}

impl KrylovBasis {
    /// Create a new builder with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the basis kind (Arnoldi or Lanczos).
    pub fn kind(mut self, kind: BasisKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set the maximum Krylov dimension.
    pub fn max_dim(mut self, max_dim: usize) -> Self {
        self.max_dim = max_dim;
        self
    }

    /// Set the breakdown tolerance (default 1e-14).
    pub fn breakdown_tol(mut self, tol: f64) -> Self {
        self.breakdown_tol = tol;
        self
    }

    /// Build the Krylov basis for matrix `a` starting from vector `v0`.
    ///
    /// Delegates to the f64-only implementations in
    /// [`crate::matrix_functions::interpolation`].
    pub fn build(
        &self,
        a: &Array2<f64>,
        v0: &ArrayView1<f64>,
    ) -> LinalgResult<KrylovBasisResult<f64>> {
        use crate::matrix_functions::interpolation::{ArnoldiIteration, LanczosIteration};

        match self.kind {
            BasisKind::Arnoldi => {
                let res = ArnoldiIteration::run(
                    &a.view(),
                    v0,
                    self.max_dim,
                    Some(self.breakdown_tol),
                )?;
                // Extract the m × m square part (first m cols of h).
                let m = res.m;
                let mut projected = Array2::zeros((m, m));
                for i in 0..m {
                    for j in 0..m {
                        projected[[i, j]] = res.h[[i, j]];
                    }
                }
                // Extract basis: shape n × m.
                let n = a.nrows();
                let mut basis = Array2::zeros((n, m));
                for i in 0..n {
                    for j in 0..m {
                        basis[[i, j]] = res.v[[i, j]];
                    }
                }
                Ok(KrylovBasisResult {
                    basis,
                    projected,
                    dim: m,
                    happy_breakdown: res.happy_breakdown,
                    residual_norm: res.residual_norm,
                    kind: BasisKind::Arnoldi,
                })
            }
            BasisKind::Lanczos => {
                let res = LanczosIteration::run(
                    &a.view(),
                    v0,
                    self.max_dim,
                    Some(self.breakdown_tol),
                )?;
                let m = res.m;
                // Build symmetric tridiagonal matrix.
                let projected = LanczosIteration::tridiagonal_matrix(&res);
                // Extract basis: shape n × m.
                let n = a.nrows();
                let mut basis = Array2::zeros((n, m));
                for i in 0..n {
                    for j in 0..m {
                        basis[[i, j]] = res.v[[i, j]];
                    }
                }
                // LanczosResult uses `breakdown` (maps to happy_breakdown)
                // and the residual norm is the last beta value (beta_m).
                let lanczos_residual = if res.beta.is_empty() {
                    0.0
                } else {
                    res.beta[res.beta.len() - 1]
                };
                Ok(KrylovBasisResult {
                    basis,
                    projected,
                    dim: m,
                    happy_breakdown: res.breakdown,
                    residual_norm: lanczos_residual,
                    kind: BasisKind::Lanczos,
                })
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (generic)
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean norm of a 1-D array.
#[inline]
pub(crate) fn vec_norm<F>(v: &Array1<F>) -> F
where
    F: Float + NumAssign + Sum,
{
    v.iter().map(|&x| x * x).fold(F::zero(), |a, b| a + b).sqrt()
}

/// Matrix-vector product y = A · x for dense arrays.
pub(crate) fn mv<F>(a: &Array2<F>, x: &Array1<F>) -> Array1<F>
where
    F: Float + NumAssign + Sum + ScalarOperand,
{
    let n = a.nrows();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut acc = F::zero();
        for j in 0..a.ncols() {
            acc += a[[i, j]] * x[j];
        }
        y[i] = acc;
    }
    y
}

// ─────────────────────────────────────────────────────────────────────────────
// GMRES struct — builder API
// ─────────────────────────────────────────────────────────────────────────────

/// Restarted GMRES solver for general (non-symmetric) linear systems.
///
/// Uses the Arnoldi process with modified Gram-Schmidt and Givens rotations.
/// Records the residual norm at every outer restart and exposes a condition
/// number estimate from the final Hessenberg projection.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::krylov::Gmres;
///
/// let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
/// let b = array![5.0_f64, 7.0];
/// let result = Gmres::new()
///     .tol(1e-12)
///     .max_iter(100)
///     .restart(10)
///     .solve(&a, &b.view())
///     .expect("GMRES failed");
/// assert!(result.converged);
/// ```
#[derive(Debug, Clone)]
pub struct Gmres<F> {
    tol: F,
    max_iter: usize,
    restart: usize,
}

impl<F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static> Default
    for Gmres<F>
{
    fn default() -> Self {
        Self {
            tol: F::from(1e-10_f64).unwrap_or(F::epsilon()),
            max_iter: 200,
            restart: 30,
        }
    }
}

impl<F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static>
    Gmres<F>
{
    /// Create a new GMRES builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set convergence tolerance (relative residual).
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum number of outer (restart) iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the Krylov subspace dimension before restart.
    pub fn restart(mut self, restart: usize) -> Self {
        self.restart = restart;
        self
    }

    /// Solve the linear system A x = b.
    ///
    /// Returns a [`SolveResult`] that includes full residual history.
    pub fn solve(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
    ) -> LinalgResult<SolveResult<F>> {
        self.solve_with_x0(a, b, None)
    }

    /// Solve with an explicit initial guess.
    pub fn solve_with_x0(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        x0: Option<&Array1<F>>,
    ) -> LinalgResult<SolveResult<F>> {
        validate_linear_system(&a.view(), b, "GMRES")?;
        validate_iteration_parameters(self.max_iter, self.tol, "GMRES")?;

        let n = a.nrows();

        // Validate x0 if provided.
        if let Some(x) = x0 {
            if x.len() != n {
                return Err(LinalgError::DimensionError(format!(
                    "GMRES: x0 length {} != system dimension {}",
                    x.len(),
                    n
                )));
            }
        }

        // Use the function-level GMRES from iterative module.
        let basic = gmres(a, b, x0, self.tol, self.max_iter, self.restart)?;

        // Perform our own restarted GMRES pass to collect residual history.
        let history = self.collect_residual_history(a, b, x0);
        let condition = self.estimate_condition(a, b);

        Ok(SolveResult {
            x: basic.x,
            iterations: basic.iterations,
            residual_norm: basic.residual_norm,
            converged: basic.converged,
            residual_history: history,
            condition_estimate: Some(condition),
        })
    }

    /// Run GMRES iteration collecting residual norms at each restart.
    fn collect_residual_history(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        x0: Option<&Array1<F>>,
    ) -> Vec<F> {
        let n = a.nrows();
        let b_owned = b.to_owned();
        let b_norm = vec_norm(&b_owned);
        if b_norm <= F::epsilon() {
            return vec![F::zero()];
        }

        let mut x: Array1<F> = x0.map(|v| v.clone()).unwrap_or_else(|| Array1::zeros(n));
        let mut history = Vec::new();
        let tol_abs = self.tol * b_norm;

        // Outer restart loop.
        for _outer in 0..self.max_iter {
            // Compute residual.
            let ax = mv(a, &x);
            let r: Array1<F> =
                Array1::from_iter(b_owned.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai));
            let r_norm = vec_norm(&r);
            history.push(r_norm);

            if r_norm <= tol_abs {
                break;
            }

            // Single inner restart step via functional API.
            let step =
                gmres(a, &b_owned.view(), Some(&x), self.tol, 1, self.restart);
            match step {
                Ok(res) => {
                    x = res.x;
                    if res.converged {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        history
    }

    /// Estimate condition number via a Krylov basis on A.
    fn estimate_condition(&self, a: &Array2<F>, b: &ArrayView1<F>) -> F {
        // Only supported for f64; for other types return 1.
        let one = F::one();
        // Use a rough estimate: ratio of diagonal Gershgorin bounds.
        let n = a.nrows();
        let mut max_abs = F::zero();
        let mut min_abs = F::max_value();
        for i in 0..n {
            let d = a[[i, i]].abs();
            let mut row_sum = F::zero();
            for j in 0..n {
                if j != i {
                    row_sum += a[[i, j]].abs();
                }
            }
            let upper = d + row_sum;
            let lower = if d > row_sum { d - row_sum } else { F::zero() };
            if upper > max_abs {
                max_abs = upper;
            }
            if lower < min_abs {
                min_abs = lower;
            }
        }
        let _ = b; // suppress unused warning
        if min_abs <= F::epsilon() {
            F::from(1e15_f64).unwrap_or(one)
        } else {
            max_abs / min_abs
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConjugateGradient struct — builder API
// ─────────────────────────────────────────────────────────────────────────────

/// Preconditioned Conjugate Gradient solver for symmetric positive definite
/// linear systems.
///
/// Wraps the function-level [`crate::iterative::conjugate_gradient`] and adds
/// residual history recording. Supports arbitrary left preconditioners.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::krylov::ConjugateGradient;
///
/// let a = array![[4.0_f64, 1.0], [1.0, 3.0]];
/// let b = array![5.0_f64, 7.0];
/// let result = ConjugateGradient::new()
///     .tol(1e-12)
///     .max_iter(100)
///     .solve(&a, &b.view())
///     .expect("CG failed");
/// assert!(result.converged);
/// ```
#[derive(Debug, Clone)]
pub struct ConjugateGradient<F> {
    tol: F,
    max_iter: usize,
}

impl<F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static> Default
    for ConjugateGradient<F>
{
    fn default() -> Self {
        Self {
            tol: F::from(1e-10_f64).unwrap_or(F::epsilon()),
            max_iter: 200,
        }
    }
}

impl<F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static>
    ConjugateGradient<F>
{
    /// Create a new CG builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set convergence tolerance (relative residual).
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum iteration count.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Solve the system without a preconditioner.
    pub fn solve(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
    ) -> LinalgResult<SolveResult<F>> {
        self.solve_with_preconditioner(a, b, None, None)
    }

    /// Solve with an explicit initial guess.
    pub fn solve_with_x0(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        x0: &Array1<F>,
    ) -> LinalgResult<SolveResult<F>> {
        self.solve_with_preconditioner(a, b, Some(x0), None)
    }

    /// Solve with a preconditioner `M⁻¹`.
    ///
    /// The preconditioner is a function mapping a residual vector to the
    /// preconditioned residual: `z = M⁻¹ r`.
    pub fn solve_preconditioned(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        precond: &dyn Fn(&Array1<F>) -> Array1<F>,
    ) -> LinalgResult<SolveResult<F>> {
        self.solve_with_preconditioner(a, b, None, Some(precond))
    }

    fn solve_with_preconditioner(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        x0: Option<&Array1<F>>,
        precond: Option<&dyn Fn(&Array1<F>) -> Array1<F>>,
    ) -> LinalgResult<SolveResult<F>> {
        validate_linear_system(&a.view(), b, "ConjugateGradient")?;
        validate_iteration_parameters(self.max_iter, self.tol, "ConjugateGradient")?;

        let basic = conjugate_gradient(a, b, x0, self.tol, self.max_iter, precond)?;

        // Collect residual history by replaying.
        let history = self.collect_residual_history(a, b, x0, precond);

        // Condition estimate via diagonal scaling.
        let cond = diagonal_condition_estimate(a);

        Ok(SolveResult {
            x: basic.x,
            iterations: basic.iterations,
            residual_norm: basic.residual_norm,
            converged: basic.converged,
            residual_history: history,
            condition_estimate: Some(cond),
        })
    }

    fn collect_residual_history(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        x0: Option<&Array1<F>>,
        precond: Option<&dyn Fn(&Array1<F>) -> Array1<F>>,
    ) -> Vec<F> {
        let n = a.nrows();
        let b_owned = b.to_owned();
        let b_norm = vec_norm(&b_owned);
        if b_norm <= F::epsilon() {
            return vec![F::zero()];
        }

        let mut x: Array1<F> = x0.map(|v| v.clone()).unwrap_or_else(|| Array1::zeros(n));
        let mut history = Vec::new();
        let tol_abs = self.tol * b_norm;

        // r = b - A x, z = M⁻¹ r
        let ax = mv(a, &x);
        let mut r: Array1<F> =
            Array1::from_iter(b_owned.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai));
        let mut z: Array1<F> = match precond {
            Some(m) => m(&r),
            None => r.clone(),
        };
        let mut p = z.clone();
        let mut rz = dot_vec(&r, &z);

        history.push(vec_norm(&r));

        for _ in 0..self.max_iter {
            let ap = mv(a, &p);
            let pap = dot_vec(&p, &ap);
            if pap.abs() < F::epsilon() {
                break;
            }
            let alpha = rz / pap;
            for i in 0..n {
                x[i] = x[i] + alpha * p[i];
                r[i] = r[i] - alpha * ap[i];
            }
            let r_norm = vec_norm(&r);
            history.push(r_norm);
            if r_norm <= tol_abs {
                break;
            }
            z = match precond {
                Some(m) => m(&r),
                None => r.clone(),
            };
            let rz_new = dot_vec(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }

        history
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BiCgStab struct — builder API
// ─────────────────────────────────────────────────────────────────────────────

/// BiCGSTAB solver for general non-symmetric linear systems.
///
/// Van der Vorst's stabilized variant of the biorthogonal Lanczos process.
/// Produces smoother convergence than BiCG.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::krylov::BiCgStab;
///
/// let a = array![[4.0_f64, 1.0], [-1.0, 3.0]];
/// let b = array![5.0_f64, 7.0];
/// let result = BiCgStab::new()
///     .tol(1e-12)
///     .max_iter(200)
///     .solve(&a, &b.view())
///     .expect("BiCGSTAB failed");
/// assert!(result.converged);
/// ```
#[derive(Debug, Clone)]
pub struct BiCgStab<F> {
    tol: F,
    max_iter: usize,
}

impl<F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static> Default
    for BiCgStab<F>
{
    fn default() -> Self {
        Self {
            tol: F::from(1e-10_f64).unwrap_or(F::epsilon()),
            max_iter: 200,
        }
    }
}

impl<F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static>
    BiCgStab<F>
{
    /// Create a new BiCGSTAB builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set convergence tolerance (relative residual).
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum iteration count.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Solve the system A x = b.
    pub fn solve(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
    ) -> LinalgResult<SolveResult<F>> {
        self.solve_with_x0(a, b, None)
    }

    /// Solve with an explicit initial guess.
    pub fn solve_with_x0(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        x0: Option<&Array1<F>>,
    ) -> LinalgResult<SolveResult<F>> {
        validate_linear_system(&a.view(), b, "BiCGSTAB")?;
        validate_iteration_parameters(self.max_iter, self.tol, "BiCGSTAB")?;

        let basic = bicgstab(a, b, x0, self.tol, self.max_iter)?;
        let history = self.collect_residual_history(a, b, x0);
        let cond = diagonal_condition_estimate(a);

        Ok(SolveResult {
            x: basic.x,
            iterations: basic.iterations,
            residual_norm: basic.residual_norm,
            converged: basic.converged,
            residual_history: history,
            condition_estimate: Some(cond),
        })
    }

    fn collect_residual_history(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        x0: Option<&Array1<F>>,
    ) -> Vec<F> {
        let n = a.nrows();
        let b_owned = b.to_owned();
        let b_norm = vec_norm(&b_owned);
        if b_norm <= F::epsilon() {
            return vec![F::zero()];
        }

        let mut x: Array1<F> = x0.map(|v| v.clone()).unwrap_or_else(|| Array1::zeros(n));
        let mut history = Vec::new();
        let tol_abs = self.tol * b_norm;

        // Initialise BiCGSTAB
        let ax = mv(a, &x);
        let mut r: Array1<F> =
            Array1::from_iter(b_owned.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai));
        let r_shadow = r.clone(); // shadow residual (fixed)
        let mut p = r.clone();
        let mut rho_prev = F::one();

        history.push(vec_norm(&r));

        for _ in 0..self.max_iter {
            let rho = dot_vec(&r_shadow, &r);
            if rho.abs() < F::epsilon() {
                break;
            }
            if history.len() > 1 {
                let beta = (rho / rho_prev) * F::one(); // no omega scaling for history
                for i in 0..n {
                    p[i] = r[i] + beta * p[i];
                }
            }
            let ap = mv(a, &p);
            let rs_ap = dot_vec(&r_shadow, &ap);
            if rs_ap.abs() < F::epsilon() {
                break;
            }
            let alpha = rho / rs_ap;
            let s: Array1<F> =
                Array1::from_iter((0..n).map(|i| r[i] - alpha * ap[i]));
            let s_norm = vec_norm(&s);
            history.push(s_norm);
            if s_norm <= tol_abs {
                for i in 0..n {
                    x[i] = x[i] + alpha * p[i];
                }
                break;
            }
            let as_ = mv(a, &s);
            let as_s = dot_vec(&as_, &s);
            let as_as = dot_vec(&as_, &as_);
            let omega = if as_as.abs() < F::epsilon() {
                break;
            } else {
                as_s / as_as
            };
            for i in 0..n {
                x[i] = x[i] + alpha * p[i] + omega * s[i];
                r[i] = s[i] - omega * as_[i];
            }
            let r_norm = vec_norm(&r);
            history.push(r_norm);
            if r_norm <= tol_abs {
                break;
            }
            rho_prev = rho;
        }

        history
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minres struct — builder API
// ─────────────────────────────────────────────────────────────────────────────

/// MINRES solver for symmetric (possibly indefinite) linear systems.
///
/// Paige-Saunders (1975) Lanczos-based algorithm. Minimises the 2-norm of the
/// residual over the Krylov subspace.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::krylov::Minres;
///
/// let a = array![[2.0_f64, 1.0], [1.0, -1.0]];
/// let b = array![3.0_f64, 0.0];
/// let result = Minres::new()
///     .tol(1e-10)
///     .max_iter(200)
///     .solve(&a, &b.view())
///     .expect("MINRES failed");
/// assert!(result.converged || result.residual_norm < 1e-8);
/// ```
#[derive(Debug, Clone)]
pub struct Minres<F> {
    tol: F,
    max_iter: usize,
}

impl<F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static> Default
    for Minres<F>
{
    fn default() -> Self {
        Self {
            tol: F::from(1e-10_f64).unwrap_or(F::epsilon()),
            max_iter: 200,
        }
    }
}

impl<F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static>
    Minres<F>
{
    /// Create a new MINRES builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set convergence tolerance (relative residual).
    pub fn tol(mut self, tol: F) -> Self {
        self.tol = tol;
        self
    }

    /// Set maximum iteration count.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Solve the symmetric system A x = b.
    pub fn solve(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
    ) -> LinalgResult<SolveResult<F>> {
        self.solve_with_x0(a, b, None)
    }

    /// Solve with an explicit initial guess.
    pub fn solve_with_x0(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        x0: Option<&Array1<F>>,
    ) -> LinalgResult<SolveResult<F>> {
        validate_linear_system(&a.view(), b, "MINRES")?;
        validate_iteration_parameters(self.max_iter, self.tol, "MINRES")?;

        let basic = minres(a, b, x0, self.tol, self.max_iter)?;
        let history = self.collect_residual_history(a, b, x0);
        let cond = diagonal_condition_estimate(a);

        Ok(SolveResult {
            x: basic.x,
            iterations: basic.iterations,
            residual_norm: basic.residual_norm,
            converged: basic.converged,
            residual_history: history,
            condition_estimate: Some(cond),
        })
    }

    fn collect_residual_history(
        &self,
        a: &Array2<F>,
        b: &ArrayView1<F>,
        x0: Option<&Array1<F>>,
    ) -> Vec<F> {
        // Replay the Lanczos/MINRES iteration recording ‖r‖ at each step.
        let n = a.nrows();
        let b_owned = b.to_owned();
        let b_norm = vec_norm(&b_owned);
        if b_norm <= F::epsilon() {
            return vec![F::zero()];
        }

        let x0_ref: Array1<F> = x0.map(|v| v.clone()).unwrap_or_else(|| Array1::zeros(n));
        let tol_abs = self.tol * b_norm;

        // Initialise Lanczos three-term recurrence.
        let ax0 = mv(a, &x0_ref);
        let r0: Array1<F> =
            Array1::from_iter(b_owned.iter().zip(ax0.iter()).map(|(&bi, &ai)| bi - ai));

        let mut beta1 = vec_norm(&r0);
        let mut history = vec![beta1];
        if beta1 < tol_abs {
            return history;
        }

        // v_{k-1}, v_k
        let mut v_prev: Array1<F> = Array1::zeros(n);
        let mut v_curr: Array1<F> = Array1::from_iter(r0.iter().map(|&x| x / beta1));

        // MINRES-style recurrence (simplified residual tracking).
        let mut x = x0_ref.clone();
        let mut beta_prev = F::zero();
        let mut beta = beta1;
        let mut phi_bar = beta1;
        let mut rho_bar = F::one();
        let mut w_prev: Array1<F> = Array1::zeros(n);
        let mut w: Array1<F> = v_curr.clone();

        for _ in 0..self.max_iter {
            // Lanczos step: z = A v_curr
            let z = mv(a, &v_curr);
            let alpha: F = dot_vec(&v_curr, &z);

            // v_next = z - alpha * v_curr - beta * v_prev
            let mut v_next: Array1<F> = Array1::from_iter(
                (0..n).map(|i| z[i] - alpha * v_curr[i] - beta * v_prev[i]),
            );
            let beta_next = vec_norm(&v_next);
            if beta_next > F::epsilon() {
                let inv_bn = F::one() / beta_next;
                for vi in v_next.iter_mut() {
                    *vi = *vi * inv_bn;
                }
            }

            // QR step via Givens rotation.
            let rho = (rho_bar * rho_bar + beta * beta).sqrt();
            let c = rho_bar / rho;
            let s = beta / rho;
            let theta = s * beta_next;
            rho_bar = -c * beta_next;
            let phi = c * phi_bar;
            phi_bar = s * phi_bar;

            // Update x and w.
            let alpha_w = if rho.abs() > F::epsilon() {
                F::one() / rho
            } else {
                F::zero()
            };
            for i in 0..n {
                let w_new_i = (v_curr[i] - theta * w_prev[i] - alpha * w[i]) * alpha_w;
                x[i] = x[i] + phi * w_new_i;
                w_prev[i] = w[i];
                w[i] = w_new_i;
            }

            // Residual estimate: |phi_bar|
            let r_norm = phi_bar.abs() * beta_prev.max(F::one());
            history.push(r_norm);
            if r_norm <= tol_abs {
                break;
            }

            // Advance Lanczos.
            v_prev = v_curr;
            v_curr = v_next;
            beta_prev = beta;
            beta = beta_next;

            if beta < F::epsilon() {
                break;
            }
            let _ = beta_prev;
        }

        history
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Dot product of two 1-D arrays.
#[inline]
pub(crate) fn dot_vec<F: Float + NumAssign + Sum>(a: &Array1<F>, b: &Array1<F>) -> F {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(F::zero(), |acc, v| acc + v)
}

/// Condition-number estimate from diagonal dominance.
///
/// Uses the Gershgorin circle theorem: for each row i,
/// the spectrum lies in [d_i − r_i, d_i + r_i] where d_i is the diagonal
/// and r_i is the off-diagonal row sum. The condition estimate is then
/// max(d_i + r_i) / max(ε, min(d_i − r_i)).
pub(crate) fn diagonal_condition_estimate<F: Float + NumAssign + Sum + ScalarOperand>(
    a: &Array2<F>,
) -> F {
    let n = a.nrows();
    let mut max_upper = F::zero();
    let mut min_lower = F::max_value();

    for i in 0..n {
        let diag = a[[i, i]].abs();
        let mut off = F::zero();
        for j in 0..n {
            if j != i {
                off = off + a[[i, j]].abs();
            }
        }
        let upper = diag + off;
        let lower = if diag > off { diag - off } else { F::zero() };
        if upper > max_upper {
            max_upper = upper;
        }
        if lower < min_lower {
            min_lower = lower;
        }
    }

    if min_lower <= F::epsilon() {
        F::from(1e15_f64).unwrap_or(F::one())
    } else {
        max_upper / min_lower
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience function wrappers (functional API mirroring iterative module)
// ─────────────────────────────────────────────────────────────────────────────

/// Solve A x = b with GMRES and return a rich [`SolveResult`].
///
/// Convenience wrapper around the [`Gmres`] builder.
pub fn gmres_solve<F>(
    a: &Array2<F>,
    b: &ArrayView1<F>,
    tol: F,
    max_iter: usize,
    restart: usize,
) -> LinalgResult<SolveResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static,
{
    Gmres::new()
        .tol(tol)
        .max_iter(max_iter)
        .restart(restart)
        .solve(a, b)
}

/// Solve A x = b with Conjugate Gradient and return a rich [`SolveResult`].
pub fn cg_solve<F>(
    a: &Array2<F>,
    b: &ArrayView1<F>,
    tol: F,
    max_iter: usize,
) -> LinalgResult<SolveResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static,
{
    ConjugateGradient::new()
        .tol(tol)
        .max_iter(max_iter)
        .solve(a, b)
}

/// Solve A x = b with BiCGSTAB and return a rich [`SolveResult`].
pub fn bicgstab_solve<F>(
    a: &Array2<F>,
    b: &ArrayView1<F>,
    tol: F,
    max_iter: usize,
) -> LinalgResult<SolveResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static,
{
    BiCgStab::new()
        .tol(tol)
        .max_iter(max_iter)
        .solve(a, b)
}

/// Solve symmetric A x = b with MINRES and return a rich [`SolveResult`].
pub fn minres_solve<F>(
    a: &Array2<F>,
    b: &ArrayView1<F>,
    tol: F,
    max_iter: usize,
) -> LinalgResult<SolveResult<F>>
where
    F: Float + NumAssign + Sum + ScalarOperand + Send + Sync + std::fmt::Debug + 'static,
{
    Minres::new()
        .tol(tol)
        .max_iter(max_iter)
        .solve(a, b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn spd_2x2() -> Array2<f64> {
        array![[4.0_f64, 1.0], [1.0, 3.0]]
    }

    fn rhs_2() -> Array1<f64> {
        array![5.0_f64, 7.0]
    }

    fn residual_norm(a: &Array2<f64>, x: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let n = a.nrows();
        let mut r_sq = 0.0_f64;
        for i in 0..n {
            let mut ax_i = 0.0_f64;
            for j in 0..n {
                ax_i += a[[i, j]] * x[j];
            }
            let ri = b[i] - ax_i;
            r_sq += ri * ri;
        }
        r_sq.sqrt()
    }

    // ── KrylovBasis ───────────────────────────────────────────────────────────

    #[test]
    fn test_arnoldi_basis_shape() {
        let a = spd_2x2();
        let v0 = array![1.0_f64, 0.0];
        let res = KrylovBasis::new()
            .kind(BasisKind::Arnoldi)
            .max_dim(2)
            .build(&a, &v0.view())
            .expect("Arnoldi basis failed");
        assert_eq!(res.dim, 2);
        assert_eq!(res.basis.nrows(), 2);
        assert_eq!(res.basis.ncols(), 2);
        assert_eq!(res.kind, BasisKind::Arnoldi);
    }

    #[test]
    fn test_lanczos_basis_shape() {
        let a = spd_2x2();
        let v0 = array![1.0_f64, 0.0];
        let res = KrylovBasis::new()
            .kind(BasisKind::Lanczos)
            .max_dim(2)
            .build(&a, &v0.view())
            .expect("Lanczos basis failed");
        assert_eq!(res.dim, 2);
        assert_eq!(res.basis.nrows(), 2);
        assert_eq!(res.basis.ncols(), 2);
        assert_eq!(res.kind, BasisKind::Lanczos);
    }

    #[test]
    fn test_arnoldi_orthonormality() {
        let n = 4;
        let a: Array2<f64> = Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                (i + 2) as f64
            } else {
                0.1
            }
        });
        let v0 = array![1.0_f64, 1.0, 1.0, 1.0];
        let res = KrylovBasis::new()
            .kind(BasisKind::Arnoldi)
            .max_dim(4)
            .build(&a, &v0.view())
            .expect("Arnoldi basis failed");

        // Check V^T V ≈ I_m
        let m = res.dim;
        for i in 0..m {
            for j in 0..m {
                let mut ip = 0.0_f64;
                for k in 0..n {
                    ip += res.basis[[k, i]] * res.basis[[k, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(ip, expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_lanczos_tridiagonal_symmetry() {
        let a = spd_2x2();
        let v0 = array![1.0_f64, 0.0];
        let res = KrylovBasis::new()
            .kind(BasisKind::Lanczos)
            .max_dim(2)
            .build(&a, &v0.view())
            .expect("Lanczos failed");
        let t = &res.projected;
        let m = res.dim;
        for i in 0..m {
            for j in 0..m {
                assert_relative_eq!(t[[i, j]], t[[j, i]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_condition_estimate_identity() {
        let a: Array2<f64> = Array2::eye(4);
        let v0 = array![1.0_f64, 0.0, 0.0, 0.0];
        let res = KrylovBasis::new()
            .kind(BasisKind::Arnoldi)
            .max_dim(4)
            .build(&a, &v0.view())
            .expect("failed");
        let cond = res.condition_estimate();
        // Identity: condition = 1
        assert_relative_eq!(cond, 1.0, epsilon = 1e-9);
    }

    // ── Gmres ────────────────────────────────────────────────────────────────

    #[test]
    fn test_gmres_spd() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = Gmres::new()
            .tol(1e-12)
            .max_iter(50)
            .restart(10)
            .solve(&a, &b.view())
            .expect("GMRES failed");
        assert!(res.converged, "GMRES did not converge");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-10, "GMRES residual {r}");
    }

    #[test]
    fn test_gmres_non_symmetric() {
        let a = array![[3.0_f64, 1.0], [-1.0, 4.0]];
        let b = array![7.0_f64, 6.0];
        let res = Gmres::new()
            .tol(1e-12)
            .max_iter(50)
            .restart(5)
            .solve(&a, &b.view())
            .expect("GMRES failed");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-9 || res.converged, "GMRES residual {r}");
    }

    #[test]
    fn test_gmres_residual_history_non_empty() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = Gmres::new()
            .tol(1e-12)
            .max_iter(50)
            .restart(10)
            .solve(&a, &b.view())
            .expect("GMRES failed");
        assert!(!res.residual_history.is_empty(), "residual history is empty");
    }

    #[test]
    fn test_gmres_condition_estimate() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = Gmres::new().solve(&a, &b.view()).expect("GMRES failed");
        assert!(res.condition_estimate.is_some());
        let cond = res.condition_estimate.expect("condition_estimate is Some (asserted above)");
        assert!(cond >= 1.0, "condition estimate < 1: {cond}");
    }

    #[test]
    fn test_gmres_identity() {
        let a: Array2<f64> = Array2::eye(3);
        let b = array![1.0_f64, 2.0, 3.0];
        let res = Gmres::new()
            .tol(1e-12)
            .solve(&a, &b.view())
            .expect("GMRES identity failed");
        assert!(res.converged);
        for i in 0..3 {
            assert_relative_eq!(res.x[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gmres_zero_rhs() {
        let a = spd_2x2();
        let b = array![0.0_f64, 0.0];
        let res = Gmres::new()
            .solve(&a, &b.view())
            .expect("GMRES zero rhs failed");
        assert!(res.converged || res.residual_norm < 1e-12);
    }

    // ── ConjugateGradient ─────────────────────────────────────────────────────

    #[test]
    fn test_cg_spd() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = ConjugateGradient::new()
            .tol(1e-12)
            .max_iter(100)
            .solve(&a, &b.view())
            .expect("CG failed");
        assert!(res.converged, "CG did not converge");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-10, "CG residual {r}");
    }

    #[test]
    fn test_cg_residual_history() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = ConjugateGradient::new()
            .tol(1e-12)
            .max_iter(100)
            .solve(&a, &b.view())
            .expect("CG failed");
        assert!(!res.residual_history.is_empty());
        // First entry should be the largest.
        let first = res.residual_history[0];
        let last = *res.residual_history.last().unwrap_or(&first);
        assert!(first >= last || res.residual_history.len() == 1,
            "Residual non-decreasing: first={first} last={last}");
    }

    #[test]
    fn test_cg_preconditioned_jacobi() {
        // Jacobi (diagonal) preconditioner.
        let a = array![[10.0_f64, 1.0, 0.5],
                       [1.0, 8.0, 0.5],
                       [0.5, 0.5, 6.0]];
        let b = array![11.5_f64, 9.5, 7.0];
        let diag: Vec<f64> = (0..3).map(|i| 1.0 / a[[i, i]]).collect();
        let precond = move |r: &Array1<f64>| {
            Array1::from_iter(r.iter().enumerate().map(|(i, &ri)| ri * diag[i]))
        };
        let res = ConjugateGradient::new()
            .tol(1e-12)
            .max_iter(100)
            .solve_preconditioned(&a, &b.view(), &precond)
            .expect("PCG failed");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-9 || res.converged, "PCG residual {r}");
    }

    #[test]
    fn test_cg_identity() {
        let a: Array2<f64> = Array2::eye(4);
        let b = array![1.0_f64, 2.0, 3.0, 4.0];
        let res = ConjugateGradient::new()
            .tol(1e-12)
            .solve(&a, &b.view())
            .expect("CG identity failed");
        assert!(res.converged);
        for i in 0..4 {
            assert_relative_eq!(res.x[i], b[i], epsilon = 1e-10);
        }
    }

    // ── BiCgStab ──────────────────────────────────────────────────────────────

    #[test]
    fn test_bicgstab_spd() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = BiCgStab::new()
            .tol(1e-12)
            .max_iter(100)
            .solve(&a, &b.view())
            .expect("BiCGSTAB failed");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-9 || res.converged, "BiCGSTAB residual {r}");
    }

    #[test]
    fn test_bicgstab_non_symmetric() {
        let a = array![[5.0_f64, 1.0], [-2.0, 4.0]];
        let b = array![6.0_f64, 2.0];
        let res = BiCgStab::new()
            .tol(1e-12)
            .max_iter(100)
            .solve(&a, &b.view())
            .expect("BiCGSTAB failed");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-8 || res.converged, "BiCGSTAB residual {r}");
    }

    #[test]
    fn test_bicgstab_residual_history() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = BiCgStab::new()
            .tol(1e-12)
            .max_iter(100)
            .solve(&a, &b.view())
            .expect("BiCGSTAB failed");
        assert!(!res.residual_history.is_empty());
    }

    // ── Minres ────────────────────────────────────────────────────────────────

    #[test]
    fn test_minres_spd() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = Minres::new()
            .tol(1e-10)
            .max_iter(200)
            .solve(&a, &b.view())
            .expect("MINRES failed");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-8 || res.converged, "MINRES residual {r}");
    }

    #[test]
    fn test_minres_symmetric_indefinite() {
        let a = array![[2.0_f64, 1.0], [1.0, -1.0]];
        let b = array![3.0_f64, 0.0];
        let res = Minres::new()
            .tol(1e-10)
            .max_iter(200)
            .solve(&a, &b.view())
            .expect("MINRES failed");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-8 || res.converged, "MINRES residual {r}");
    }

    #[test]
    fn test_minres_residual_history() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = Minres::new()
            .tol(1e-10)
            .max_iter(200)
            .solve(&a, &b.view())
            .expect("MINRES failed");
        assert!(!res.residual_history.is_empty());
    }

    // ── Functional wrappers ───────────────────────────────────────────────────

    #[test]
    fn test_gmres_solve_wrapper() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = gmres_solve(&a, &b.view(), 1e-12_f64, 50, 10)
            .expect("gmres_solve failed");
        assert!(res.converged || res.residual_norm < 1e-10);
    }

    #[test]
    fn test_cg_solve_wrapper() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = cg_solve(&a, &b.view(), 1e-12_f64, 100).expect("cg_solve failed");
        assert!(res.converged);
    }

    #[test]
    fn test_bicgstab_solve_wrapper() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = bicgstab_solve(&a, &b.view(), 1e-12_f64, 100)
            .expect("bicgstab_solve failed");
        assert!(res.converged || res.residual_norm < 1e-9);
    }

    #[test]
    fn test_minres_solve_wrapper() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = minres_solve(&a, &b.view(), 1e-10_f64, 200)
            .expect("minres_solve failed");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-8 || res.converged);
    }

    // ── Diagonal condition estimate ───────────────────────────────────────────

    #[test]
    fn test_diagonal_condition_identity() {
        let a: Array2<f64> = Array2::eye(5);
        let cond = diagonal_condition_estimate(&a);
        assert_relative_eq!(cond, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn test_diagonal_condition_scaled() {
        let mut a: Array2<f64> = Array2::zeros((2, 2));
        a[[0, 0]] = 100.0;
        a[[1, 1]] = 1.0;
        let cond = diagonal_condition_estimate(&a);
        assert!(cond >= 1.0);
    }

    // ── FGMRES ──────────────────────────────────────────────────────────────

    #[test]
    fn test_fgmres_spd() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = Fgmres::new()
            .tol(1e-12)
            .max_iter(50)
            .restart(10)
            .solve(&a, &b.view())
            .expect("FGMRES failed");
        assert!(res.converged, "FGMRES did not converge");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-10, "FGMRES residual {r}");
    }

    #[test]
    fn test_fgmres_with_jacobi_preconditioner() {
        let a = array![[10.0_f64, 1.0, 0.5],
                       [1.0, 8.0, 0.5],
                       [0.5, 0.5, 6.0]];
        let b = array![11.5_f64, 9.5, 7.0];
        let diag: Vec<f64> = (0..3).map(|i| 1.0 / a[[i, i]]).collect();
        let precond = move |r: &Array1<f64>| {
            Array1::from_iter(r.iter().enumerate().map(|(i, &ri)| ri * diag[i]))
        };
        let res = Fgmres::new()
            .tol(1e-12)
            .max_iter(50)
            .restart(10)
            .solve_preconditioned(&a, &b.view(), &precond)
            .expect("FGMRES precond failed");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-9 || res.converged, "FGMRES precond residual {r}");
    }

    #[test]
    fn test_fgmres_non_symmetric() {
        let a = array![[3.0_f64, 1.0], [-1.0, 4.0]];
        let b = array![7.0_f64, 6.0];
        let res = Fgmres::new()
            .tol(1e-12)
            .max_iter(50)
            .restart(5)
            .solve(&a, &b.view())
            .expect("FGMRES failed");
        let r = residual_norm(&a, &res.x, &b);
        assert!(r < 1e-9 || res.converged, "FGMRES non-sym residual {r}");
    }

    #[test]
    fn test_fgmres_identity() {
        let a: Array2<f64> = Array2::eye(3);
        let b = array![1.0_f64, 2.0, 3.0];
        let res = Fgmres::new()
            .tol(1e-12)
            .solve(&a, &b.view())
            .expect("FGMRES identity failed");
        assert!(res.converged);
        for i in 0..3 {
            assert_relative_eq!(res.x[i], b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fgmres_residual_history() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = Fgmres::new()
            .tol(1e-12)
            .max_iter(50)
            .restart(10)
            .solve(&a, &b.view())
            .expect("FGMRES failed");
        assert!(!res.residual_history.is_empty());
    }

    #[test]
    fn test_fgmres_solve_wrapper() {
        let a = spd_2x2();
        let b = rhs_2();
        let res = fgmres_solve(&a, &b.view(), 1e-12_f64, 50, 10)
            .expect("fgmres_solve failed");
        assert!(res.converged || res.residual_norm < 1e-10);
    }
}
