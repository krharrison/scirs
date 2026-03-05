//! Enhanced Spectral Methods for PDE solving
//!
//! Provides Fourier and Chebyshev spectral methods for PDEs with
//! spectral differentiation matrices, dealiasing, and pseudospectral
//! collocation.
//!
//! ## Features
//! - Fourier spectral method for periodic problems (heat, advection)
//! - Chebyshev spectral method for non-periodic problems
//! - Spectral differentiation matrices (first and second derivative)
//! - Dealiasing via the 2/3 rule
//! - Pseudospectral collocation for nonlinear problems
//! - High accuracy for smooth solutions (exponential convergence)

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use std::f64::consts::PI;

use crate::ode::{solve_ivp, ODEMethod, ODEOptions};
use crate::pde::{PDEError, PDEResult};

// ---------------------------------------------------------------------------
// Spectral basis types
// ---------------------------------------------------------------------------

/// Spectral basis type for the enhanced solver
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralBasisType {
    /// Fourier basis (periodic problems)
    Fourier,
    /// Chebyshev basis (non-periodic problems)
    Chebyshev,
}

/// Options for spectral PDE solvers
#[derive(Debug, Clone)]
pub struct SpectralEnhancedOptions {
    /// Number of collocation points / modes
    pub n_modes: usize,
    /// Whether to apply 2/3 dealiasing rule
    pub dealias: bool,
    /// Absolute ODE tolerance
    pub atol: f64,
    /// Relative ODE tolerance
    pub rtol: f64,
    /// Max ODE steps
    pub max_steps: usize,
}

impl Default for SpectralEnhancedOptions {
    fn default() -> Self {
        SpectralEnhancedOptions {
            n_modes: 64,
            dealias: true,
            atol: 1e-8,
            rtol: 1e-6,
            max_steps: 10000,
        }
    }
}

/// Result from spectral solve
#[derive(Debug, Clone)]
pub struct SpectralEnhancedResult {
    /// Collocation points
    pub x: Array1<f64>,
    /// Time points
    pub t: Vec<f64>,
    /// Solution snapshots: `u[step]` is array of values at collocation points
    pub u: Vec<Array1<f64>>,
    /// Spectral coefficients at final time
    pub coefficients: Array1<f64>,
}

// ---------------------------------------------------------------------------
// Fourier spectral differentiation
// ---------------------------------------------------------------------------

/// Compute the Fourier spectral first-derivative matrix for N points on [0, L)
///
/// For periodic functions sampled at equally spaced points x_j = j*L/N.
/// `D[i,j]` approximates d/dx evaluated at x_i from values at x_j.
pub fn fourier_diff_matrix(n: usize, domain_length: f64) -> PDEResult<Array2<f64>> {
    if n < 3 {
        return Err(PDEError::SpectralError(
            "Need at least 3 Fourier points".to_string(),
        ));
    }
    if !n.is_multiple_of(2) {
        return Err(PDEError::SpectralError(
            "Number of Fourier points should be even".to_string(),
        ));
    }

    let h = domain_length / n as f64;
    let mut d = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let diff = (i as f64 - j as f64) * PI / n as f64;
                if (i as isize - j as isize).unsigned_abs().is_multiple_of(2) {
                    d[[i, j]] = 0.0; // cancels for even separation when N is even
                } else {
                    d[[i, j]] = 0.5 / (diff.tan()) * (2.0 * PI / domain_length);
                    // Correct sign
                    if (i as isize - j as isize) > 0 && (i as isize - j as isize) % 2 != 0 {
                        d[[i, j]] = -d[[i, j]].abs();
                        // Re-derive properly
                    }
                }
            }
        }
    }

    // Use the proper Fourier differentiation matrix formula
    // D_{ij} = (pi/L) * (-1)^{i+j} * cot(pi*(i-j)/N)  for i != j
    // D_{ii} = 0
    let mut d2 = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                let angle = PI * (i as f64 - j as f64) / n as f64;
                let cot_val = angle.cos() / angle.sin();
                d2[[i, j]] = sign * PI / domain_length * cot_val;
            }
        }
    }

    Ok(d2)
}

/// Compute the Fourier spectral second-derivative matrix for N points on [0, L)
pub fn fourier_diff2_matrix(n: usize, domain_length: f64) -> PDEResult<Array2<f64>> {
    if n < 3 {
        return Err(PDEError::SpectralError(
            "Need at least 3 Fourier points".to_string(),
        ));
    }
    if !n.is_multiple_of(2) {
        return Err(PDEError::SpectralError(
            "Number of Fourier points should be even".to_string(),
        ));
    }

    let mut d2 = Array2::zeros((n, n));
    let scale = (2.0 * PI / domain_length) * (2.0 * PI / domain_length);

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                let angle = PI * (i as f64 - j as f64) / n as f64;
                let sin_a = angle.sin();
                let val = -sign / (2.0 * sin_a * sin_a);
                d2[[i, j]] = val * scale;
            }
        }
        // Diagonal: D2_{ii} = -(N^2 + 2) / 12 * (2*pi/L)^2
        // For the standard formula on [0, 2*pi):
        let n_f = n as f64;
        d2[[i, i]] = -(n_f * n_f + 2.0) / 12.0 * scale;
    }

    Ok(d2)
}

// ---------------------------------------------------------------------------
// Chebyshev spectral differentiation
// ---------------------------------------------------------------------------

/// Compute Chebyshev collocation points on [-1, 1]
///
/// x_j = cos(j * pi / N), j = 0, 1, ..., N
/// Returns N+1 points (including endpoints).
pub fn chebyshev_collocation_points(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n + 1, |j| (j as f64 * PI / n as f64).cos())
}

/// Compute Chebyshev spectral first-derivative matrix (N+1 x N+1)
///
/// Based on the standard Chebyshev differentiation matrix.
/// Points are x_j = cos(j*pi/N) for j = 0..N.
pub fn chebyshev_diff_matrix(n: usize) -> PDEResult<Array2<f64>> {
    if n < 2 {
        return Err(PDEError::SpectralError(
            "Need at least 2 for Chebyshev differentiation".to_string(),
        ));
    }

    let n1 = n + 1;
    let x = chebyshev_collocation_points(n);
    let mut d = Array2::zeros((n1, n1));

    // Chebyshev weight factors
    let mut c = vec![1.0; n1];
    c[0] = 2.0;
    c[n] = 2.0;

    for i in 0..n1 {
        for j in 0..n1 {
            if i != j {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                d[[i, j]] = (c[i] / c[j]) * sign / (x[i] - x[j]);
            }
        }
    }

    // Diagonal entries: D[i,i] = -sum_{j!=i} D[i,j]
    for i in 0..n1 {
        let mut sum = 0.0;
        for j in 0..n1 {
            if j != i {
                sum += d[[i, j]];
            }
        }
        d[[i, i]] = -sum;
    }

    Ok(d)
}

/// Compute Chebyshev second-derivative matrix by squaring the first-derivative matrix
pub fn chebyshev_diff2_matrix(n: usize) -> PDEResult<Array2<f64>> {
    let d1 = chebyshev_diff_matrix(n)?;
    Ok(d1.dot(&d1))
}

/// Map Chebyshev points from [-1,1] to [a, b]
pub fn map_chebyshev_to_interval(x_cheb: &Array1<f64>, a: f64, b: f64) -> Array1<f64> {
    let half = (b - a) / 2.0;
    let mid = (a + b) / 2.0;
    Array1::from_shape_fn(x_cheb.len(), |i| mid + half * x_cheb[i])
}

/// Scale Chebyshev differentiation matrix from \[-1,1\] to \[a,b\]
///
/// On \[a,b\], the derivative scales by 2/(b-a).
pub fn scale_chebyshev_diff(d: &Array2<f64>, a: f64, b: f64) -> Array2<f64> {
    let scale = 2.0 / (b - a);
    d.mapv(|v| v * scale)
}

// ---------------------------------------------------------------------------
// Dealiasing (2/3 rule)
// ---------------------------------------------------------------------------

/// Apply the 2/3 dealiasing rule to Fourier coefficients.
///
/// Zeros out the top 1/3 of the frequency spectrum to prevent aliasing
/// errors in quadratic nonlinearities.
pub fn dealias_23(coefficients: &mut Array1<f64>) {
    let n = coefficients.len();
    let cutoff = 2 * n / 3;
    for i in cutoff..n {
        coefficients[i] = 0.0;
    }
}

/// Apply dealiasing to a 2D array of Fourier coefficients (rows are modes)
pub fn dealias_23_2d(coefficients: &mut Array2<f64>) {
    let n_rows = coefficients.shape()[0];
    let n_cols = coefficients.shape()[1];
    let cutoff_r = 2 * n_rows / 3;
    let cutoff_c = 2 * n_cols / 3;

    for i in cutoff_r..n_rows {
        for j in 0..n_cols {
            coefficients[[i, j]] = 0.0;
        }
    }
    for i in 0..n_rows {
        for j in cutoff_c..n_cols {
            coefficients[[i, j]] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Fourier pseudospectral solver for periodic problems
// ---------------------------------------------------------------------------

/// Solve 1D periodic diffusion: du/dt = alpha * d2u/dx2
/// using Fourier pseudospectral method.
///
/// The spatial domain is [0, L) with periodic boundary conditions.
/// Exponential convergence for smooth periodic solutions.
pub fn fourier_diffusion_1d(
    alpha: f64,
    domain_length: f64,
    t_range: [f64; 2],
    initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
    options: &SpectralEnhancedOptions,
) -> PDEResult<SpectralEnhancedResult> {
    let n = options.n_modes;
    if n < 4 || !n.is_multiple_of(2) {
        return Err(PDEError::SpectralError(
            "Number of modes must be even and >= 4".to_string(),
        ));
    }

    // Collocation points
    let x = Array1::from_shape_fn(n, |i| i as f64 * domain_length / n as f64);

    // Second derivative matrix
    let d2 = fourier_diff2_matrix(n, domain_length)?;

    // Initial condition
    let u0 = Array1::from_shape_fn(n, |i| initial_condition(x[i]));

    // ODE RHS: du/dt = alpha * D2 * u
    let rhs = move |_t: f64, u: ArrayView1<f64>| -> Array1<f64> {
        let mut dudt = Array1::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += d2[[i, j]] * u[j];
            }
            dudt[i] = alpha * sum;
        }
        dudt
    };

    let ode_opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: options.rtol,
        atol: options.atol,
        max_steps: options.max_steps,
        dense_output: false,
        ..Default::default()
    };

    let rhs_arc = std::sync::Arc::new(rhs);
    let rhs_clone = move |t: f64, u: ArrayView1<f64>| -> Array1<f64> { rhs_arc(t, u) };
    let result = solve_ivp(rhs_clone, t_range, u0.clone(), Some(ode_opts))?;

    let t_vec: Vec<f64> = result.t.to_vec();
    let u_vec: Vec<Array1<f64>> = result.y.to_vec();
    let coeffs = if let Some(last) = u_vec.last() {
        last.clone()
    } else {
        u0
    };

    Ok(SpectralEnhancedResult {
        x,
        t: t_vec,
        u: u_vec,
        coefficients: coeffs,
    })
}

/// Solve 1D periodic advection: du/dt + c * du/dx = 0
/// using Fourier pseudospectral method.
pub fn fourier_advection_1d(
    velocity: f64,
    domain_length: f64,
    t_range: [f64; 2],
    initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
    options: &SpectralEnhancedOptions,
) -> PDEResult<SpectralEnhancedResult> {
    let n = options.n_modes;
    if n < 4 || !n.is_multiple_of(2) {
        return Err(PDEError::SpectralError(
            "Number of modes must be even and >= 4".to_string(),
        ));
    }

    let x = Array1::from_shape_fn(n, |i| i as f64 * domain_length / n as f64);
    let d1 = fourier_diff_matrix(n, domain_length)?;

    let u0 = Array1::from_shape_fn(n, |i| initial_condition(x[i]));

    let rhs = move |_t: f64, u: ArrayView1<f64>| -> Array1<f64> {
        let mut dudt = Array1::zeros(n);
        for i in 0..n {
            let mut du_dx = 0.0;
            for j in 0..n {
                du_dx += d1[[i, j]] * u[j];
            }
            dudt[i] = -velocity * du_dx;
        }
        dudt
    };

    let ode_opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: options.rtol,
        atol: options.atol,
        max_steps: options.max_steps,
        dense_output: false,
        ..Default::default()
    };

    let rhs_arc = std::sync::Arc::new(rhs);
    let rhs_clone = move |t: f64, u: ArrayView1<f64>| -> Array1<f64> { rhs_arc(t, u) };
    let result = solve_ivp(rhs_clone, t_range, u0.clone(), Some(ode_opts))?;

    let t_vec: Vec<f64> = result.t.to_vec();
    let u_vec: Vec<Array1<f64>> = result.y.to_vec();
    let coeffs = u_vec.last().cloned().unwrap_or(u0);

    Ok(SpectralEnhancedResult {
        x,
        t: t_vec,
        u: u_vec,
        coefficients: coeffs,
    })
}

// ---------------------------------------------------------------------------
// Chebyshev pseudospectral solver for non-periodic problems
// ---------------------------------------------------------------------------

/// Solve 1D Poisson equation: d2u/dx2 = f(x) on [a, b]
/// using Chebyshev pseudospectral method.
///
/// Boundary conditions: u(a) = ua, u(b) = ub (Dirichlet).
pub fn chebyshev_poisson_1d(
    source: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    n: usize,
    u_a: f64,
    u_b: f64,
) -> PDEResult<SpectralEnhancedResult> {
    if n < 4 {
        return Err(PDEError::SpectralError(
            "Need at least 4 Chebyshev points".to_string(),
        ));
    }

    // Chebyshev points on [-1,1] and mapped to [a,b]
    let x_cheb = chebyshev_collocation_points(n);
    let x_phys = map_chebyshev_to_interval(&x_cheb, a, b);

    // Second derivative matrix on [a,b]
    let d2_ref = chebyshev_diff2_matrix(n)?;
    let scale = 2.0 / (b - a);
    let d2 = d2_ref.mapv(|v| v * scale * scale);

    let n1 = n + 1;
    // Interior points: 1..n (indices in Chebyshev array, x_0 and x_n are boundaries)
    let interior = n1 - 2;

    // Build system for interior: D2[1..n-1, 1..n-1] * u_int = f_int - BC contributions
    let mut a_mat = Array2::zeros((interior, interior));
    let mut rhs = Array1::zeros(interior);

    for i in 0..interior {
        let gi = i + 1; // global index
        for j in 0..interior {
            let gj = j + 1;
            a_mat[[i, j]] = d2[[gi, gj]];
        }
        rhs[i] = source(x_phys[gi]) - d2[[gi, 0]] * u_b - d2[[gi, n]] * u_a;
        // Note: Chebyshev points: x_0=cos(0)=1 maps to b, x_n=cos(pi)=-1 maps to a
    }

    // Solve
    let u_int = solve_dense(&a_mat, &rhs)?;

    // Assemble full solution
    let mut u = Array1::zeros(n1);
    u[0] = u_b; // x_0 = b (cos(0)=1 maps to b)
    u[n] = u_a; // x_n = a (cos(pi)=-1 maps to a)
    for i in 0..interior {
        u[i + 1] = u_int[i];
    }

    Ok(SpectralEnhancedResult {
        x: x_phys,
        t: vec![0.0],
        u: vec![u.clone()],
        coefficients: u,
    })
}

/// Solve 1D diffusion on [a, b] with Chebyshev spectral method:
///   du/dt = alpha * d2u/dx2
/// with Dirichlet BCs: u(a,t) = u_a, u(b,t) = u_b.
pub fn chebyshev_diffusion_1d(
    alpha: f64,
    a: f64,
    b: f64,
    t_range: [f64; 2],
    n: usize,
    initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
    u_a: f64,
    u_b: f64,
    options: &SpectralEnhancedOptions,
) -> PDEResult<SpectralEnhancedResult> {
    if n < 4 {
        return Err(PDEError::SpectralError(
            "Need at least 4 Chebyshev points".to_string(),
        ));
    }

    let x_cheb = chebyshev_collocation_points(n);
    let x_phys = map_chebyshev_to_interval(&x_cheb, a, b);

    let d2_ref = chebyshev_diff2_matrix(n)?;
    let scale = 2.0 / (b - a);
    let d2 = d2_ref.mapv(|v| v * scale * scale);

    let n1 = n + 1;
    let mut u0 = Array1::from_shape_fn(n1, |i| initial_condition(x_phys[i]));
    u0[0] = u_b; // Chebyshev x[0] = cos(0) = 1 maps to b
    u0[n] = u_a; // Chebyshev x[n] = cos(pi) = -1 maps to a

    let rhs = move |_t: f64, u: ArrayView1<f64>| -> Array1<f64> {
        let mut dudt = Array1::zeros(n1);
        for i in 1..n1 - 1 {
            let mut sum = 0.0;
            for j in 0..n1 {
                sum += d2[[i, j]] * u[j];
            }
            dudt[i] = alpha * sum;
        }
        // Boundary: du/dt = 0 (Dirichlet held fixed)
        dudt[0] = 0.0;
        dudt[n1 - 1] = 0.0;
        dudt
    };

    let ode_opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: options.rtol,
        atol: options.atol,
        max_steps: options.max_steps,
        dense_output: false,
        ..Default::default()
    };

    let rhs_arc = std::sync::Arc::new(rhs);
    let rhs_clone = move |t: f64, u: ArrayView1<f64>| -> Array1<f64> { rhs_arc(t, u) };
    let result = solve_ivp(rhs_clone, t_range, u0.clone(), Some(ode_opts))?;

    let t_vec: Vec<f64> = result.t.to_vec();
    let u_vec: Vec<Array1<f64>> = result.y.to_vec();
    let coeffs = u_vec.last().cloned().unwrap_or(u0);

    Ok(SpectralEnhancedResult {
        x: x_phys,
        t: t_vec,
        u: u_vec,
        coefficients: coeffs,
    })
}

// ---------------------------------------------------------------------------
// Pseudospectral collocation for nonlinear problems
// ---------------------------------------------------------------------------

/// Solve nonlinear 1D BVP via Chebyshev collocation + Newton iteration:
///   d2u/dx2 = F(x, u, du/dx)
/// with u(a) = u_a, u(b) = u_b.
///
/// Uses Newton iteration with numerical Jacobian.
pub fn chebyshev_nonlinear_bvp(
    nonlinear_rhs: &dyn Fn(f64, f64, f64) -> f64, // F(x, u, du/dx)
    a: f64,
    b: f64,
    n: usize,
    u_a: f64,
    u_b: f64,
    initial_guess: &dyn Fn(f64) -> f64,
    max_newton_iter: usize,
    newton_tol: f64,
) -> PDEResult<SpectralEnhancedResult> {
    if n < 4 {
        return Err(PDEError::SpectralError(
            "Need at least 4 Chebyshev points".to_string(),
        ));
    }

    let x_cheb = chebyshev_collocation_points(n);
    let x_phys = map_chebyshev_to_interval(&x_cheb, a, b);
    let n1 = n + 1;

    let d1_ref = chebyshev_diff_matrix(n)?;
    let d2_ref = chebyshev_diff2_matrix(n)?;
    let scale = 2.0 / (b - a);
    let d1 = d1_ref.mapv(|v| v * scale);
    let d2 = d2_ref.mapv(|v| v * scale * scale);

    // Interior indices: 1..n-1
    let interior = n1 - 2;

    // Initial guess
    let mut u = Array1::from_shape_fn(n1, |i| initial_guess(x_phys[i]));
    u[0] = u_b;
    u[n] = u_a;

    for _iter in 0..max_newton_iter {
        // Compute du/dx and d2u/dx2
        let du_dx = d1.dot(&u);
        let d2u_dx2 = d2.dot(&u);

        // Residual: R_i = d2u/dx2[i] - F(x_i, u_i, du_dx_i) for interior points
        let mut residual = Array1::zeros(interior);
        for i in 0..interior {
            let gi = i + 1;
            residual[i] = d2u_dx2[gi] - nonlinear_rhs(x_phys[gi], u[gi], du_dx[gi]);
        }

        let res_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        if res_norm < newton_tol {
            break;
        }

        // Jacobian via finite differences (numerical)
        let eps = 1e-8;
        let mut jac = Array2::zeros((interior, interior));
        for j in 0..interior {
            let gj = j + 1;
            let mut u_pert = u.clone();
            u_pert[gj] += eps;

            let du_pert = d1.dot(&u_pert);
            let d2u_pert = d2.dot(&u_pert);

            for i in 0..interior {
                let gi = i + 1;
                let r_pert = d2u_pert[gi] - nonlinear_rhs(x_phys[gi], u_pert[gi], du_pert[gi]);
                jac[[i, j]] = (r_pert - residual[i]) / eps;
            }
        }

        // Solve J * delta = -R
        let neg_res = residual.mapv(|v| -v);
        let delta = solve_dense(&jac, &neg_res)?;

        for i in 0..interior {
            u[i + 1] += delta[i];
        }
    }

    Ok(SpectralEnhancedResult {
        x: x_phys,
        t: vec![0.0],
        u: vec![u.clone()],
        coefficients: u,
    })
}

// ---------------------------------------------------------------------------
// Dense linear solver (shared helper)
// ---------------------------------------------------------------------------

/// Solve dense linear system Ax = b
fn solve_dense(a: &Array2<f64>, b: &Array1<f64>) -> PDEResult<Array1<f64>> {
    let n = b.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let mut a_c = a.clone();
    let mut b_c = b.clone();

    for k in 0..n {
        let mut max_val = a_c[[k, k]].abs();
        let mut max_row = k;
        for i in k + 1..n {
            let val = a_c[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }
        if max_val < 1e-14 {
            return Err(PDEError::ComputationError(
                "Singular matrix in spectral solve".to_string(),
            ));
        }
        if max_row != k {
            for j in k..n {
                let tmp = a_c[[k, j]];
                a_c[[k, j]] = a_c[[max_row, j]];
                a_c[[max_row, j]] = tmp;
            }
            let tmp = b_c[k];
            b_c[k] = b_c[max_row];
            b_c[max_row] = tmp;
        }
        for i in k + 1..n {
            let factor = a_c[[i, k]] / a_c[[k, k]];
            for j in k + 1..n {
                a_c[[i, j]] -= factor * a_c[[k, j]];
            }
            b_c[i] -= factor * b_c[k];
        }
    }

    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i + 1..n {
            sum += a_c[[i, j]] * x[j];
        }
        x[i] = (b_c[i] - sum) / a_c[[i, i]];
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_points() {
        let x = chebyshev_collocation_points(4);
        assert_eq!(x.len(), 5);
        assert!((x[0] - 1.0).abs() < 1e-14);
        assert!((x[4] - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_chebyshev_diff_constant() {
        // Derivative of constant should be zero
        let n = 8;
        let d = chebyshev_diff_matrix(n).expect("Should create diff matrix");
        let u = Array1::ones(n + 1);
        let du = d.dot(&u);
        for &v in du.iter() {
            assert!(
                v.abs() < 1e-10,
                "Derivative of constant should be 0, got {v}"
            );
        }
    }

    #[test]
    fn test_chebyshev_diff_linear() {
        // d/dx(x) = 1 on [-1,1]
        let n = 8;
        let d = chebyshev_diff_matrix(n).expect("diff matrix");
        let x = chebyshev_collocation_points(n);
        let du = d.dot(&x);
        for i in 0..n + 1 {
            assert!(
                (du[i] - 1.0).abs() < 1e-10,
                "d/dx(x) at x={}: got {}, expected 1.0",
                x[i],
                du[i]
            );
        }
    }

    #[test]
    fn test_chebyshev_diff_quadratic() {
        // d/dx(x^2) = 2x
        let n = 8;
        let d = chebyshev_diff_matrix(n).expect("diff matrix");
        let x = chebyshev_collocation_points(n);
        let u = x.mapv(|v| v * v);
        let du = d.dot(&u);
        for i in 0..n + 1 {
            let expected = 2.0 * x[i];
            assert!(
                (du[i] - expected).abs() < 1e-8,
                "d/dx(x^2) at x={}: got {}, expected {expected}",
                x[i],
                du[i]
            );
        }
    }

    #[test]
    fn test_chebyshev_diff2_quadratic() {
        // d2/dx2(x^2) = 2
        let n = 10;
        let d2 = chebyshev_diff2_matrix(n).expect("diff2 matrix");
        let x = chebyshev_collocation_points(n);
        let u = x.mapv(|v| v * v);
        let d2u = d2.dot(&u);
        for i in 1..n {
            assert!(
                (d2u[i] - 2.0).abs() < 1e-6,
                "d2/dx2(x^2) at x={}: got {}, expected 2.0",
                x[i],
                d2u[i]
            );
        }
    }

    #[test]
    fn test_chebyshev_poisson_1d() {
        // d2u/dx2 = -pi^2 * sin(pi*x) on [0,1], u(0)=0, u(1)=0
        // Exact: u = sin(pi*x)
        let result = chebyshev_poisson_1d(&|x| -PI * PI * (PI * x).sin(), 0.0, 1.0, 20, 0.0, 0.0)
            .expect("Should succeed");

        for i in 0..result.x.len() {
            let exact = (PI * result.x[i]).sin();
            assert!(
                (result.u[0][i] - exact).abs() < 0.05,
                "Chebyshev Poisson at x={:.3}: got {:.4}, expected {:.4}",
                result.x[i],
                result.u[0][i],
                exact
            );
        }
    }

    #[test]
    fn test_chebyshev_diffusion_decay() {
        // du/dt = alpha * d2u/dx2, u(x,0)=sin(pi*x), u(0)=0, u(1)=0
        let alpha = 0.1;
        let result = chebyshev_diffusion_1d(
            alpha,
            0.0,
            1.0,
            [0.0, 0.5],
            16,
            |x| (PI * x).sin(),
            0.0,
            0.0,
            &SpectralEnhancedOptions {
                n_modes: 16,
                atol: 1e-8,
                rtol: 1e-6,
                ..Default::default()
            },
        )
        .expect("Should succeed");

        let last = &result.u[result.u.len() - 1];
        // Find point nearest to x=0.5 (interior Chebyshev point near 0.5)
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for i in 0..result.x.len() {
            let d = (result.x[i] - 0.5).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        let exact = (PI * result.x[best_idx]).sin() * (-PI * PI * alpha * 0.5).exp();
        assert!(
            (last[best_idx] - exact).abs() < 0.1,
            "Chebyshev diffusion near x=0.5: got {}, expected {exact}",
            last[best_idx]
        );
    }

    #[test]
    fn test_fourier_diff_matrix_constant() {
        let n = 8;
        let d = fourier_diff_matrix(n, 2.0 * PI).expect("diff matrix");
        let u = Array1::ones(n);
        let du = d.dot(&u);
        for &v in du.iter() {
            assert!(
                v.abs() < 1e-10,
                "Fourier derivative of constant should be 0, got {v}"
            );
        }
    }

    #[test]
    fn test_fourier_diff2_matrix_sine() {
        // d2/dx2(sin(x)) = -sin(x) on [0, 2*pi)
        let n = 16;
        let l = 2.0 * PI;
        let d2 = fourier_diff2_matrix(n, l).expect("diff2 matrix");
        let x = Array1::from_shape_fn(n, |i| i as f64 * l / n as f64);
        let u = x.mapv(|v| v.sin());
        let d2u = d2.dot(&u);
        for i in 0..n {
            let expected = -x[i].sin();
            assert!(
                (d2u[i] - expected).abs() < 0.1,
                "Fourier d2/dx2(sin(x)) at x={:.3}: got {:.4}, expected {:.4}",
                x[i],
                d2u[i],
                expected
            );
        }
    }

    #[test]
    fn test_dealias_23() {
        let mut coeffs = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        dealias_23(&mut coeffs);
        // Cutoff = 2*6/3 = 4, so indices 4,5 zeroed
        assert!((coeffs[0] - 1.0).abs() < 1e-15);
        assert!((coeffs[3] - 4.0).abs() < 1e-15);
        assert!((coeffs[4]).abs() < 1e-15);
        assert!((coeffs[5]).abs() < 1e-15);
    }

    #[test]
    fn test_fourier_diffusion_1d() {
        let alpha = 0.01;
        let result = fourier_diffusion_1d(
            alpha,
            2.0 * PI,
            [0.0, 1.0],
            |x| x.sin(),
            &SpectralEnhancedOptions {
                n_modes: 16,
                ..Default::default()
            },
        )
        .expect("Should succeed");

        assert!(result.u.len() > 1);
        // Solution should decay: max amplitude at final time < initial
        let u0_max: f64 = result.u[0].iter().copied().fold(0.0_f64, f64::max);
        let u_final_max: f64 = result.u[result.u.len() - 1]
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        assert!(
            u_final_max <= u0_max + 1e-6,
            "Solution should decay: initial max={u0_max}, final max={u_final_max}"
        );
    }

    #[test]
    fn test_fourier_advection_1d() {
        let result = fourier_advection_1d(
            1.0,
            2.0 * PI,
            [0.0, 1.0],
            |x| x.sin(),
            &SpectralEnhancedOptions {
                n_modes: 16,
                ..Default::default()
            },
        )
        .expect("Should succeed");

        assert!(result.u.len() > 1);
    }

    #[test]
    fn test_chebyshev_map_interval() {
        let x = chebyshev_collocation_points(4);
        let mapped = map_chebyshev_to_interval(&x, 0.0, 1.0);
        // x[0]=1 -> 1.0, x[4]=-1 -> 0.0
        assert!((mapped[0] - 1.0).abs() < 1e-14);
        assert!((mapped[4] - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_chebyshev_nonlinear_bvp() {
        // Solve u'' = 0, u(0)=0, u(1)=1 => u=x (trivially linear)
        let result = chebyshev_nonlinear_bvp(
            &|_x, _u, _du| 0.0,
            0.0,
            1.0,
            8,
            0.0,
            1.0,
            &|x| x, // good initial guess
            10,
            1e-10,
        )
        .expect("Should succeed");

        for i in 0..result.x.len() {
            assert!(
                (result.u[0][i] - result.x[i]).abs() < 1e-6,
                "NL BVP at x={:.3}: got {:.4}, expected {:.4}",
                result.x[i],
                result.u[0][i],
                result.x[i]
            );
        }
    }

    #[test]
    fn test_scale_chebyshev_diff() {
        let d = chebyshev_diff_matrix(4).expect("diff matrix");
        let d_scaled = scale_chebyshev_diff(&d, 0.0, 2.0);
        // Scale factor = 2/(2-0) = 1.0, so scaled should equal original
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (d_scaled[[i, j]] - d[[i, j]]).abs() < 1e-14,
                    "Scale 1.0 should be identity"
                );
            }
        }
    }

    #[test]
    fn test_dealias_2d() {
        let mut c = Array2::ones((6, 6));
        dealias_23_2d(&mut c);
        // Rows 4,5 and cols 4,5 should be zero
        assert!((c[[0, 0]] - 1.0).abs() < 1e-15);
        assert!((c[[3, 3]] - 1.0).abs() < 1e-15);
        assert!(c[[4, 0]].abs() < 1e-15);
        assert!(c[[0, 4]].abs() < 1e-15);
        assert!(c[[5, 5]].abs() < 1e-15);
    }
}
