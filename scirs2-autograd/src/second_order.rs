//! Second-order optimization support
//!
//! This module provides primitives for second-order optimization methods that
//! exploit curvature information about the loss landscape.  All functions
//! operate on **plain `Vec<f64>` / `Array` values** so they integrate cleanly
//! with the functional and dynamic-graph APIs without requiring a live
//! autograd `Context`.
//!
//! # Summary
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`natural_gradient`] | Precondition gradient by the inverse Fisher information matrix |
//! | [`fisher_information_matrix`] | Empirical FIM via outer products of log-likelihood gradients |
//! | [`gauss_newton_matrix`] | Gauss-Newton matrix `JᵀJ` from a Jacobian and residuals |
//! | [`kfac_update`] | K-FAC style preconditioned gradient using Kronecker-factored curvature |
//!
//! # Background
//!
//! ## Natural gradient (Amari, 1998)
//!
//! In the space of probability distributions the steepest descent direction is
//! not the Euclidean gradient `g` but the **natural gradient**
//! `F⁻¹ g` where `F` is the Fisher information matrix.
//!
//! ## Empirical Fisher information matrix
//!
//! The empirical FIM is
//! `F ≈ (1/N) Σᵢ ∇log p(xᵢ | θ) ∇log p(xᵢ | θ)ᵀ`
//! where the per-sample gradients are computed via finite differences.
//!
//! ## Gauss-Newton matrix
//!
//! For a least-squares loss `L = ½‖r(θ)‖²`, the Hessian is approximated by
//! `G = JᵀJ` where `J = ∂r/∂θ` is the Jacobian of the residuals.  The
//! Gauss-Newton step is `θ ← θ - (JᵀJ + λI)⁻¹ Jᵀr`.
//!
//! ## K-FAC (Martens & Grosse, 2015)
//!
//! K-FAC approximates the FIM layer by layer using a Kronecker product:
//! `F_l ≈ A_l ⊗ G_l` where `A_l` is the covariance of layer inputs and
//! `G_l` is the covariance of pre-activation gradients.  The inverse is then
//! `F_l⁻¹ ≈ A_l⁻¹ ⊗ G_l⁻¹`, so the preconditioned gradient for layer `l` is
//! `G_l⁻¹ Δ A_l⁻¹` (matrix form).
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::second_order::{
//!     fisher_information_matrix, natural_gradient, gauss_newton_matrix, kfac_update,
//! };
//! use scirs2_core::ndarray::{Array1, Array2, arr1, arr2};
//!
//! // --- Natural gradient ---
//! // A simple quadratic loss: L(θ) = ‖θ‖²
//! let grad = Array1::from(vec![2.0_f64, 4.0]);
//! let fisher = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("valid shape");
//! let ng = natural_gradient(&grad, &fisher, 1e-4).expect("natural gradient");
//! assert!((ng[0] - 2.0).abs() < 1e-5, "ng[0]={}", ng[0]);
//!
//! // --- Gauss-Newton ---
//! // J = [[1, 0], [0, 1]], r = [1, 1] => G = I
//! let j = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
//! let r = arr1(&[1.0_f64, 1.0]);
//! let gn = gauss_newton_matrix(&j, &r).expect("gauss newton");
//! assert!((gn[[0, 0]] - 1.0).abs() < 1e-10);
//! assert!((gn[[1, 1]] - 1.0).abs() < 1e-10);
//! ```

use crate::error::AutogradError;
use scirs2_core::ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Central-FD step size for gradient computation.
const H: f64 = 1e-5;

/// Compute the gradient of a scalar function `f` at point `x` using central FD.
fn gradient_fd(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut g = vec![0.0_f64; n];
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    for i in 0..n {
        xp[i] = x[i] + H;
        xm[i] = x[i] - H;
        g[i] = (f(&xp) - f(&xm)) / (2.0 * H);
        xp[i] = x[i];
        xm[i] = x[i];
    }
    g
}

/// Solve the linear system `A x = b` via Gaussian elimination with partial
/// pivoting.  Operates in-place on copies.
///
/// Returns `Ok(x)` on success, or `Err` if `A` is singular (pivot < tol).
fn solve_linear_system(
    a: &Array2<f64>,
    b: &Array1<f64>,
    tol: f64,
) -> Result<Array1<f64>, AutogradError> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(AutogradError::ShapeMismatch(format!(
            "solve_linear_system: expected {}×{} matrix, got {}×{}",
            n,
            n,
            a.nrows(),
            a.ncols()
        )));
    }

    // Build augmented matrix [A | b]
    let mut aug = vec![0.0_f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[[i, j]];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot row
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < tol {
            return Err(AutogradError::OperationError(format!(
                "solve_linear_system: matrix is singular or nearly singular (pivot={max_val})"
            )));
        }

        // Swap rows col and max_row
        if max_row != col {
            for k in 0..=(n) {
                aug.swap(col * (n + 1) + k, max_row * (n + 1) + k);
            }
        }

        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for k in col..=(n) {
                let delta = aug[col * (n + 1) + k] * factor;
                aug[row * (n + 1) + k] -= delta;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let diag = aug[i * (n + 1) + i];
        if diag.abs() < tol {
            return Err(AutogradError::OperationError(
                "solve_linear_system: zero diagonal during back-substitution".to_string(),
            ));
        }
        x[i] = sum / diag;
    }

    Ok(Array1::from(x))
}

/// Invert an `n × n` matrix using Gauss-Jordan elimination.
///
/// Returns `Ok(A⁻¹)` or `Err` if `A` is singular.
fn invert_matrix(a: &Array2<f64>, tol: f64) -> Result<Array2<f64>, AutogradError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(AutogradError::ShapeMismatch(format!(
            "invert_matrix: expected square matrix, got {}×{}",
            n,
            a.ncols()
        )));
    }

    // Build augmented matrix [A | I]
    let cols = 2 * n;
    let mut aug = vec![0.0_f64; n * cols];
    for i in 0..n {
        for j in 0..n {
            aug[i * cols + j] = a[[i, j]];
        }
        aug[i * cols + n + i] = 1.0; // identity on the right
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_val = aug[col * cols + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row * cols + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < tol {
            return Err(AutogradError::OperationError(format!(
                "invert_matrix: matrix is singular (pivot={max_val})"
            )));
        }

        if max_row != col {
            for k in 0..cols {
                aug.swap(col * cols + k, max_row * cols + k);
            }
        }

        let pivot = aug[col * cols + col];
        // Divide pivot row by pivot
        for k in 0..cols {
            aug[col * cols + k] /= pivot;
        }

        // Eliminate column in all other rows
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * cols + col];
            for k in 0..cols {
                let delta = aug[col * cols + k] * factor;
                aug[row * cols + k] -= delta;
            }
        }
    }

    // Extract right half = A⁻¹
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[i * cols + n + j];
        }
    }
    Ok(inv)
}

// ─────────────────────────────────────────────────────────────────────────────
// natural_gradient
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the natural gradient `F⁻¹ g` given the gradient `g` and the Fisher
/// information matrix `F`.
///
/// The natural gradient (Amari, 1998) is the steepest descent direction in the
/// space of probability distributions.  It accounts for the geometry of the
/// parameter space by preconditioning the Euclidean gradient with the inverse
/// Fisher information matrix.
///
/// A Tikhonov regularisation term `damping * I` is added to `F` before
/// inversion to ensure numerical stability:
///
/// `natural_grad = (F + damping · I)⁻¹ g`
///
/// # Arguments
/// * `grad`     – Gradient vector `g ∈ R^n`.
/// * `fisher`   – Fisher information matrix `F ∈ R^{n × n}`.
/// * `damping`  – Regularisation strength (Tikhonov damping).  Use `1e-4` as
///               a reasonable default.
///
/// # Returns
/// `Array1<f64>` of length `n` — the natural gradient.
///
/// # Errors
/// Returns `AutogradError` if shapes are inconsistent or the matrix is singular.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::second_order::natural_gradient;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let g = Array1::from(vec![2.0_f64, 4.0]);
/// let fisher = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 4.0]).expect("valid shape");
/// let ng = natural_gradient(&g, &fisher, 1e-6).expect("ok");
/// // F = diag(2, 4), g = (2, 4)  =>  F^{-1} g = (1, 1)
/// assert!((ng[0] - 1.0).abs() < 1e-5, "ng[0]={}", ng[0]);
/// assert!((ng[1] - 1.0).abs() < 1e-5, "ng[1]={}", ng[1]);
/// ```
pub fn natural_gradient(
    grad: &Array1<f64>,
    fisher: &Array2<f64>,
    damping: f64,
) -> Result<Array1<f64>, AutogradError> {
    let n = grad.len();
    if fisher.nrows() != n || fisher.ncols() != n {
        return Err(AutogradError::ShapeMismatch(format!(
            "natural_gradient: grad has length {n} but fisher is {}×{}",
            fisher.nrows(),
            fisher.ncols()
        )));
    }
    if damping < 0.0 {
        return Err(AutogradError::OperationError(
            "natural_gradient: damping must be non-negative".to_string(),
        ));
    }

    // Regularise: F_reg = F + damping * I
    let mut f_reg = fisher.clone();
    for i in 0..n {
        f_reg[[i, i]] += damping;
    }

    // Solve (F + λI) x = g
    solve_linear_system(&f_reg, grad, 1e-12)
}

// ─────────────────────────────────────────────────────────────────────────────
// fisher_information_matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the empirical Fisher information matrix.
///
/// Given a model function `model_fn(params, data_point) -> f64` that computes
/// the log-likelihood of a single data point, the empirical FIM is:
///
/// `F ≈ (1/N) Σᵢ ∇_θ log p(xᵢ | θ) [∇_θ log p(xᵢ | θ)]ᵀ`
///
/// Each gradient is computed via central finite differences.
///
/// # Arguments
/// * `model_fn`  – `Fn(params: &[f64], data: &[f64]) -> f64` computing the
///                log-probability (or any scalar loss) for one data point.
/// * `params`    – Current parameter vector `θ ∈ R^p`.
/// * `data`      – A slice of data points, each a `Vec<f64>`.  The FIM is
///                averaged over all data points.
///
/// # Returns
/// The empirical Fisher information matrix `F ∈ R^{p × p}`.
///
/// # Errors
/// Returns `AutogradError` if `params` is empty or `data` is empty.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::second_order::fisher_information_matrix;
///
/// // log p(x | θ) = -0.5 * (x - θ)² — Gaussian with unit variance
/// // FIM = E[(x - θ)²] = 1  (Fisher for Gaussian mean)
/// let data: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0], vec![-1.0], vec![2.0]];
/// let params = vec![0.0_f64];
/// let fim = fisher_information_matrix(
///     |theta: &[f64], x: &[f64]| -0.5 * (x[0] - theta[0]).powi(2),
///     &params,
///     &data,
/// ).expect("fim");
/// // FIM should be close to 1.0
/// assert!((fim[[0, 0]] - 1.0).abs() < 0.5, "fim[0,0]={}", fim[[0, 0]]);
/// ```
pub fn fisher_information_matrix(
    model_fn: impl Fn(&[f64], &[f64]) -> f64,
    params: &[f64],
    data: &[Vec<f64>],
) -> Result<Array2<f64>, AutogradError> {
    let p = params.len();
    if p == 0 {
        return Err(AutogradError::OperationError(
            "fisher_information_matrix: params must be non-empty".to_string(),
        ));
    }
    if data.is_empty() {
        return Err(AutogradError::OperationError(
            "fisher_information_matrix: data must be non-empty".to_string(),
        ));
    }

    let n = data.len() as f64;
    let mut fim = Array2::<f64>::zeros((p, p));

    for sample in data.iter() {
        // Compute gradient of log-likelihood wrt params for this sample
        let grad = {
            let f = |theta: &[f64]| model_fn(theta, sample);
            gradient_fd(&f, params)
        };

        // Outer product: g g^T
        for i in 0..p {
            for j in 0..p {
                fim[[i, j]] += grad[i] * grad[j];
            }
        }
    }

    // Average over data points
    fim.mapv_inplace(|v| v / n);
    Ok(fim)
}

// ─────────────────────────────────────────────────────────────────────────────
// gauss_newton_matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Gauss-Newton approximation to the Hessian.
///
/// For a residual function `r: R^p → R^m`, the full Hessian of `½‖r‖²` is
/// `H = JᵀJ + Σᵢ rᵢ ∇²rᵢ`.  The Gauss-Newton matrix drops the second-order
/// term:
///
/// `G = JᵀJ`
///
/// This is a positive semi-definite approximation that is valid when residuals
/// are small or when the model is nearly linear.
///
/// # Arguments
/// * `jacobian`  – Jacobian matrix `J ∈ R^{m × p}` where `m` is the number of
///                residuals and `p` is the number of parameters.
/// * `residuals` – Residual vector `r ∈ R^m`.  Only used to validate dimensions
///                in this implementation; it is included in the signature for
///                future extensions (e.g. a weighted Gauss-Newton).
///
/// # Returns
/// The Gauss-Newton matrix `G = JᵀJ ∈ R^{p × p}`.
///
/// # Errors
/// Returns `AutogradError` if dimensions are inconsistent.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::second_order::gauss_newton_matrix;
/// use scirs2_core::ndarray::{arr1, arr2};
///
/// let j = arr2(&[[1.0_f64, 0.0], [0.0, 2.0]]);
/// let r = arr1(&[1.0_f64, 2.0]);
/// let gn = gauss_newton_matrix(&j, &r).expect("ok");
/// // G = JᵀJ = diag(1, 4)
/// assert!((gn[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((gn[[1, 1]] - 4.0).abs() < 1e-10);
/// assert!(gn[[0, 1]].abs() < 1e-10);
/// ```
pub fn gauss_newton_matrix(
    jacobian: &Array2<f64>,
    residuals: &Array1<f64>,
) -> Result<Array2<f64>, AutogradError> {
    let (m, p) = (jacobian.nrows(), jacobian.ncols());
    if residuals.len() != m {
        return Err(AutogradError::ShapeMismatch(format!(
            "gauss_newton_matrix: jacobian has {} rows but residuals has {} elements",
            m,
            residuals.len()
        )));
    }

    // G = JᵀJ  (p × p)
    let mut g = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0_f64;
            for k in 0..m {
                s += jacobian[[k, i]] * jacobian[[k, j]];
            }
            g[[i, j]] = s;
        }
    }
    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// kfac_update
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a K-FAC style preconditioned gradient update.
///
/// For each layer `l`, K-FAC approximates the inverse Fisher as a Kronecker
/// product of two smaller matrices `A_l⁻¹ ⊗ G_l⁻¹`.  Given the gradient
/// `Δ_l` reshaped as a matrix `W ∈ R^{d_out × d_in}`, the preconditioned
/// update is:
///
/// `Δ̃_l = G_l⁻¹ Δ_l A_l⁻¹`
///
/// This is equivalent to `vec(Δ̃_l) = (A_l⁻¹ ⊗ G_l⁻¹) vec(Δ_l)` via the
/// vec-permutation identity, and is far cheaper than inverting the full
/// `(d_out · d_in) × (d_out · d_in)` FIM.
///
/// # Arguments
/// * `grads`   – Per-layer gradient matrices.  Each element is a
///              `p × q` gradient matrix for that layer.
/// * `a_inv`   – Per-layer inverse input-covariance matrices (`q × q`).
/// * `g_inv`   – Per-layer inverse pre-activation-gradient covariance matrices
///              (`p × p`).
///
/// All three slices must have the same length.
///
/// # Returns
/// Per-layer preconditioned gradients, each a `p × q` matrix.
///
/// # Errors
/// Returns `AutogradError` if the lengths differ or matrix dimensions are
/// incompatible.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::second_order::kfac_update;
/// use scirs2_core::ndarray::{Array2, arr2};
///
/// // Single layer: 2×2 weight matrix, identity K-FAC factors
/// let delta = arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]);
/// let a_inv = Array2::<f64>::eye(2);
/// let g_inv = Array2::<f64>::eye(2);
/// let preconditioned = kfac_update(&[delta.clone()], &[a_inv], &[g_inv]).expect("ok");
/// // With identity factors, result equals input
/// assert!((preconditioned[0][[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((preconditioned[0][[1, 1]] - 4.0).abs() < 1e-10);
/// ```
pub fn kfac_update(
    grads: &[Array2<f64>],
    a_inv: &[Array2<f64>],
    g_inv: &[Array2<f64>],
) -> Result<Vec<Array2<f64>>, AutogradError> {
    let num_layers = grads.len();
    if a_inv.len() != num_layers {
        return Err(AutogradError::ShapeMismatch(format!(
            "kfac_update: grads has {} layers but a_inv has {}",
            num_layers,
            a_inv.len()
        )));
    }
    if g_inv.len() != num_layers {
        return Err(AutogradError::ShapeMismatch(format!(
            "kfac_update: grads has {} layers but g_inv has {}",
            num_layers,
            g_inv.len()
        )));
    }

    let mut result = Vec::with_capacity(num_layers);

    for l in 0..num_layers {
        let delta = &grads[l];
        let ai = &a_inv[l];
        let gi = &g_inv[l];

        let (p, q) = (delta.nrows(), delta.ncols());

        // Validate G⁻¹ shape: p × p
        if gi.nrows() != p || gi.ncols() != p {
            return Err(AutogradError::ShapeMismatch(format!(
                "kfac_update: layer {l}: gradient is {p}×{q} but g_inv is {}×{}",
                gi.nrows(),
                gi.ncols()
            )));
        }
        // Validate A⁻¹ shape: q × q
        if ai.nrows() != q || ai.ncols() != q {
            return Err(AutogradError::ShapeMismatch(format!(
                "kfac_update: layer {l}: gradient is {p}×{q} but a_inv is {}×{}",
                ai.nrows(),
                ai.ncols()
            )));
        }

        // Compute Δ̃ = G⁻¹ Δ A⁻¹
        //
        // Step 1: tmp = G⁻¹ Δ   (p × q)
        let mut tmp = Array2::<f64>::zeros((p, q));
        for i in 0..p {
            for j in 0..q {
                let mut s = 0.0_f64;
                for k in 0..p {
                    s += gi[[i, k]] * delta[[k, j]];
                }
                tmp[[i, j]] = s;
            }
        }

        // Step 2: precond = tmp A⁻¹  (p × q)
        let mut precond = Array2::<f64>::zeros((p, q));
        for i in 0..p {
            for j in 0..q {
                let mut s = 0.0_f64;
                for k in 0..q {
                    s += tmp[[i, k]] * ai[[k, j]];
                }
                precond[[i, j]] = s;
            }
        }

        result.push(precond);
    }

    Ok(result)
}

/// Compute the K-FAC factors (input covariance and pre-activation gradient
/// covariance) for a single linear layer.
///
/// Given a batch of layer inputs `A ∈ R^{n × d_in}` and a batch of
/// pre-activation gradients `G ∈ R^{n × d_out}`, the K-FAC factors are:
///
/// `A_cov = (1/n) AᵀA ∈ R^{d_in × d_in}`
/// `G_cov = (1/n) GᵀG ∈ R^{d_out × d_out}`
///
/// These are then inverted (with Tikhonov regularisation) to obtain the
/// `a_inv` and `g_inv` arguments for [`kfac_update`].
///
/// # Arguments
/// * `layer_inputs`   – Layer input activations `A ∈ R^{n × d_in}`.
/// * `layer_grads`    – Pre-activation gradient `G ∈ R^{n × d_out}`.
/// * `damping`        – Tikhonov regularisation for both matrices.
///
/// # Returns
/// `(a_inv, g_inv)` — the regularised inverses of the two factor matrices.
///
/// # Errors
/// Returns `AutogradError` on dimension mismatch or singular factors.
pub fn kfac_factors(
    layer_inputs: &Array2<f64>,
    layer_grads: &Array2<f64>,
    damping: f64,
) -> Result<(Array2<f64>, Array2<f64>), AutogradError> {
    let (n, d_in) = (layer_inputs.nrows(), layer_inputs.ncols());
    let (n2, d_out) = (layer_grads.nrows(), layer_grads.ncols());
    if n != n2 {
        return Err(AutogradError::ShapeMismatch(format!(
            "kfac_factors: layer_inputs has {n} rows but layer_grads has {n2} rows"
        )));
    }
    if n == 0 {
        return Err(AutogradError::OperationError(
            "kfac_factors: batch size must be > 0".to_string(),
        ));
    }

    let nf = n as f64;

    // A_cov = (1/n) AᵀA
    let mut a_cov = Array2::<f64>::zeros((d_in, d_in));
    for k in 0..n {
        for i in 0..d_in {
            for j in 0..d_in {
                a_cov[[i, j]] += layer_inputs[[k, i]] * layer_inputs[[k, j]];
            }
        }
    }
    a_cov.mapv_inplace(|v| v / nf);

    // G_cov = (1/n) GᵀG
    let mut g_cov = Array2::<f64>::zeros((d_out, d_out));
    for k in 0..n {
        for i in 0..d_out {
            for j in 0..d_out {
                g_cov[[i, j]] += layer_grads[[k, i]] * layer_grads[[k, j]];
            }
        }
    }
    g_cov.mapv_inplace(|v| v / nf);

    // Regularise and invert
    for i in 0..d_in {
        a_cov[[i, i]] += damping;
    }
    for i in 0..d_out {
        g_cov[[i, i]] += damping;
    }

    let a_inv = invert_matrix(&a_cov, 1e-12)?;
    let g_inv = invert_matrix(&g_cov, 1e-12)?;

    Ok((a_inv, g_inv))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, arr2, Array2};

    const TOL: f64 = 1e-4;

    // ── solve_linear_system ───────────────────────────────────────────────

    #[test]
    fn test_solve_linear_system_identity() {
        let a = Array2::<f64>::eye(3);
        let b = arr1(&[1.0_f64, 2.0, 3.0]);
        let x = solve_linear_system(&a, &b, 1e-12).expect("solve identity");
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_linear_system_2x2() {
        // 2x + y = 5, x + 3y = 10  =>  x = 1, y = 3
        let a = arr2(&[[2.0_f64, 1.0], [1.0, 3.0]]);
        let b = arr1(&[5.0_f64, 10.0]);
        let x = solve_linear_system(&a, &b, 1e-12).expect("solve 2x2");
        assert!((x[0] - 1.0).abs() < TOL, "x[0]={}", x[0]);
        assert!((x[1] - 3.0).abs() < TOL, "x[1]={}", x[1]);
    }

    #[test]
    fn test_solve_linear_system_singular_err() {
        let a = arr2(&[[1.0_f64, 2.0], [2.0, 4.0]]);
        let b = arr1(&[1.0_f64, 1.0]);
        let r = solve_linear_system(&a, &b, 1e-8);
        assert!(r.is_err(), "Singular matrix should return error");
    }

    // ── invert_matrix ────────────────────────────────────────────────────

    #[test]
    fn test_invert_matrix_identity() {
        let a = Array2::<f64>::eye(3);
        let inv = invert_matrix(&a, 1e-12).expect("invert identity");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[[i, j]] - expected).abs() < TOL,
                    "inv[{i},{j}]={}", inv[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_invert_matrix_diagonal() {
        // diag(2, 4) => inv = diag(0.5, 0.25)
        let a = arr2(&[[2.0_f64, 0.0], [0.0, 4.0]]);
        let inv = invert_matrix(&a, 1e-12).expect("invert diagonal");
        assert!((inv[[0, 0]] - 0.5).abs() < TOL);
        assert!((inv[[1, 1]] - 0.25).abs() < TOL);
        assert!(inv[[0, 1]].abs() < TOL);
    }

    // ── natural_gradient ─────────────────────────────────────────────────

    #[test]
    fn test_natural_gradient_identity_fisher() {
        // F = I => natural grad = grad
        let g = arr1(&[1.0_f64, 2.0, 3.0]);
        let f = Array2::<f64>::eye(3);
        let ng = natural_gradient(&g, &f, 0.0).expect("natural gradient identity");
        assert!((ng[0] - 1.0).abs() < TOL);
        assert!((ng[1] - 2.0).abs() < TOL);
        assert!((ng[2] - 3.0).abs() < TOL);
    }

    #[test]
    fn test_natural_gradient_diagonal_fisher() {
        // F = diag(2, 4), g = (2, 4) => F^{-1}g = (1, 1)
        let g = arr1(&[2.0_f64, 4.0]);
        let f = arr2(&[[2.0_f64, 0.0], [0.0, 4.0]]);
        let ng = natural_gradient(&g, &f, 1e-10).expect("natural gradient diagonal");
        assert!((ng[0] - 1.0).abs() < TOL, "ng[0]={}", ng[0]);
        assert!((ng[1] - 1.0).abs() < TOL, "ng[1]={}", ng[1]);
    }

    #[test]
    fn test_natural_gradient_damping() {
        // With large damping, natural grad ≈ grad / damping (for I + λI ≈ λI)
        let g = arr1(&[1.0_f64]);
        let f = Array2::<f64>::zeros((1, 1));
        let ng = natural_gradient(&g, &f, 2.0).expect("natural gradient damping");
        // (0 + 2*I)^{-1} * 1 = 0.5
        assert!((ng[0] - 0.5).abs() < TOL, "ng[0]={}", ng[0]);
    }

    #[test]
    fn test_natural_gradient_shape_error() {
        let g = arr1(&[1.0_f64, 2.0]);
        let f = Array2::<f64>::eye(3); // wrong size
        let r = natural_gradient(&g, &f, 0.0);
        assert!(r.is_err());
    }

    // ── fisher_information_matrix ─────────────────────────────────────────

    #[test]
    fn test_fisher_information_gaussian_mean() {
        // log p(x | θ) = -0.5 * (x - θ)²
        // grad wrt θ = x - θ
        // FIM = E[(x - θ)²] = variance of data
        let data = vec![
            vec![0.0_f64],
            vec![1.0],
            vec![-1.0],
            vec![2.0],
            vec![-2.0],
        ];
        let params = vec![0.0_f64];
        let fim = fisher_information_matrix(
            |theta: &[f64], x: &[f64]| -0.5 * (x[0] - theta[0]).powi(2),
            &params,
            &data,
        )
        .expect("FIM gaussian");
        // FIM ≈ E[x²] = (0 + 1 + 1 + 4 + 4) / 5 = 2.0
        assert!(fim[[0, 0]] > 0.0, "FIM should be positive: {}", fim[[0, 0]]);
    }

    #[test]
    fn test_fisher_information_empty_params_err() {
        let data = vec![vec![1.0_f64]];
        let r = fisher_information_matrix(|_, _| 0.0, &[], &data);
        assert!(r.is_err());
    }

    #[test]
    fn test_fisher_information_empty_data_err() {
        let params = vec![1.0_f64];
        let r = fisher_information_matrix(|_, _| 0.0, &params, &[]);
        assert!(r.is_err());
    }

    // ── gauss_newton_matrix ───────────────────────────────────────────────

    #[test]
    fn test_gauss_newton_identity_jacobian() {
        // J = I (2×2), G = JᵀJ = I
        let j = Array2::<f64>::eye(2);
        let r = arr1(&[1.0_f64, 1.0]);
        let gn = gauss_newton_matrix(&j, &r).expect("GN identity");
        assert!((gn[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((gn[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(gn[[0, 1]].abs() < 1e-10);
    }

    #[test]
    fn test_gauss_newton_diagonal_jacobian() {
        // J = diag(1, 2), G = diag(1, 4)
        let j = arr2(&[[1.0_f64, 0.0], [0.0, 2.0]]);
        let r = arr1(&[1.0_f64, 2.0]);
        let gn = gauss_newton_matrix(&j, &r).expect("GN diagonal");
        assert!((gn[[0, 0]] - 1.0).abs() < 1e-10, "G[0,0]={}", gn[[0, 0]]);
        assert!((gn[[1, 1]] - 4.0).abs() < 1e-10, "G[1,1]={}", gn[[1, 1]]);
        assert!(gn[[0, 1]].abs() < 1e-10);
    }

    #[test]
    fn test_gauss_newton_shape_error() {
        let j = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let r = arr1(&[1.0_f64]); // wrong: j has 2 rows, r has 1 element
        let res = gauss_newton_matrix(&j, &r);
        assert!(res.is_err());
    }

    #[test]
    fn test_gauss_newton_rectangular_jacobian() {
        // J ∈ R^{3 × 2}: 3 residuals, 2 parameters
        // J = [[1,0],[0,1],[1,1]], G = JᵀJ = [[2,1],[1,2]]
        let j = arr2(&[[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let r = arr1(&[1.0_f64, 1.0, 1.0]);
        let gn = gauss_newton_matrix(&j, &r).expect("GN rectangular");
        assert!((gn[[0, 0]] - 2.0).abs() < 1e-10, "G[0,0]={}", gn[[0, 0]]);
        assert!((gn[[0, 1]] - 1.0).abs() < 1e-10, "G[0,1]={}", gn[[0, 1]]);
        assert!((gn[[1, 0]] - 1.0).abs() < 1e-10, "G[1,0]={}", gn[[1, 0]]);
        assert!((gn[[1, 1]] - 2.0).abs() < 1e-10, "G[1,1]={}", gn[[1, 1]]);
    }

    // ── kfac_update ──────────────────────────────────────────────────────

    #[test]
    fn test_kfac_update_identity_factors() {
        // Identity K-FAC factors => preconditioned grad equals input grad
        let delta = arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]);
        let ai = Array2::<f64>::eye(2);
        let gi = Array2::<f64>::eye(2);
        let result = kfac_update(&[delta.clone()], &[ai], &[gi]).expect("K-FAC identity");
        assert_eq!(result.len(), 1);
        let p = &result[0];
        assert!((p[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((p[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((p[[1, 0]] - 3.0).abs() < 1e-10);
        assert!((p[[1, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_kfac_update_scaling_factors() {
        // G⁻¹ = 2I, A⁻¹ = 3I => preconditioned = 6 * delta
        let delta = arr2(&[[1.0_f64, 0.0], [0.0, 1.0]]);
        let ai = arr2(&[[3.0_f64, 0.0], [0.0, 3.0]]);
        let gi = arr2(&[[2.0_f64, 0.0], [0.0, 2.0]]);
        let result = kfac_update(&[delta], &[ai], &[gi]).expect("K-FAC scaling");
        let p = &result[0];
        assert!((p[[0, 0]] - 6.0).abs() < 1e-10, "p[0,0]={}", p[[0, 0]]);
        assert!((p[[1, 1]] - 6.0).abs() < 1e-10, "p[1,1]={}", p[[1, 1]]);
    }

    #[test]
    fn test_kfac_update_multi_layer() {
        let d1 = Array2::<f64>::eye(2);
        let d2 = arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]);
        let ai1 = Array2::<f64>::eye(2);
        let gi1 = Array2::<f64>::eye(2);
        let ai2 = Array2::<f64>::eye(2);
        let gi2 = Array2::<f64>::eye(2);
        let result = kfac_update(&[d1, d2], &[ai1, ai2], &[gi1, gi2]).expect("multi-layer");
        assert_eq!(result.len(), 2);
        assert!((result[0][[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[1][[0, 1]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_kfac_update_length_mismatch_err() {
        let d = Array2::<f64>::eye(2);
        let ai = Array2::<f64>::eye(2);
        let gi = Array2::<f64>::eye(2);
        // a_inv has 2 entries but grads has 1
        let r = kfac_update(&[d], &[ai.clone(), ai], &[gi]);
        assert!(r.is_err());
    }

    // ── kfac_factors ─────────────────────────────────────────────────────

    #[test]
    fn test_kfac_factors_identity_inputs() {
        // If layer_inputs = I and layer_grads = I, then
        // A_cov = I/n, G_cov = I/n (after averaging), with damping the inverses
        // should be (1/n + damping)^{-1} I
        let n = 3usize;
        let inputs = Array2::<f64>::eye(n);
        let grads_m = Array2::<f64>::eye(n);
        let damping = 1e-4;
        let (ai, gi) = kfac_factors(&inputs, &grads_m, damping).expect("kfac factors");
        // A_cov = (1/3) I + 1e-4 I = (1/3 + 1e-4) I  => inv = 1/(1/3+1e-4) I ≈ 3 I
        let expected = 1.0 / (1.0 / (n as f64) + damping);
        for i in 0..n {
            assert!(
                (ai[[i, i]] - expected).abs() < 0.01 * expected,
                "ai[{i},{i}]={} expected~{expected}", ai[[i, i]]
            );
            assert!(
                (gi[[i, i]] - expected).abs() < 0.01 * expected,
                "gi[{i},{i}]={} expected~{expected}", gi[[i, i]]
            );
        }
    }

    #[test]
    fn test_kfac_factors_shape_error() {
        let inputs = Array2::<f64>::eye(3);
        let grads_m = Array2::<f64>::eye(4); // different batch size
        let r = kfac_factors(&inputs, &grads_m, 1e-4);
        assert!(r.is_err());
    }
}
