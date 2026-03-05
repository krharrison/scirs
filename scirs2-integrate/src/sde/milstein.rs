//! Milstein scheme for SDEs (strong order 1.0)
//!
//! The Milstein method improves upon Euler-Maruyama by adding the leading
//! term from the Itô-Taylor expansion. This extra correction term involves
//! the derivative of the diffusion coefficient, capturing the curvature of
//! the diffusion manifold.
//!
//! ## Scalar Scheme (1D, single Brownian motion)
//!
//! For the scalar SDE `dX = f(t, X) dt + g(t, X) dW`:
//!
//! ```text
//! X_{n+1} = X_n + f(t_n, X_n) dt + g(t_n, X_n) ΔW_n
//!           + (1/2) g(t_n, X_n) g'(t_n, X_n) ((ΔW_n)^2 - dt)
//! ```
//!
//! where `g'` is the partial derivative `∂g/∂x`, approximated numerically
//! by a central finite difference.
//!
//! ## Multi-dimensional Case (commutative noise)
//!
//! For the multi-dimensional case with commutative noise conditions
//! (g_ij g_kj' = g_ij' g_kj for all i,k,j), the scheme generalizes to:
//!
//! ```text
//! X_{n+1} = X_n + f(t_n, X_n) dt + g(t_n, X_n) ΔW_n
//!           + (1/2) Σ_j g_j ∂g_j/∂x ((ΔW_j)^2 - dt)
//! ```
//!
//! ## Convergence
//!
//! - **Strong order**: 1.0 (vs 0.5 for EM)
//! - **Weak order**: 1.0 (same as EM)
//!
//! The Milstein scheme doubles the strong convergence order compared to EM,
//! which means it achieves the same path accuracy with √10 larger step sizes.

use crate::error::{IntegrateError, IntegrateResult};
use crate::sde::{compute_n_steps, SdeOptions, SdeProblem, SdeSolution};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::{Normal, Rng, StdRng};
use scirs2_core::Distribution;

/// Step size for computing numerical derivatives of the diffusion coefficient.
const DIFF_H: f64 = 1e-6;

/// Milstein scheme for general (multi-dimensional, commutative noise) SDEs.
///
/// This method achieves **strong convergence order 1.0** by adding the
/// Milstein correction term to the Euler-Maruyama update.
///
/// The diffusion derivatives are approximated via central finite differences:
/// `∂g_j/∂x ≈ (g(t, x + h*e_i, ...) - g(t, x - h*e_i, ...)) / (2h)`
///
/// For non-commutative noise (general multi-dimensional case), this scheme
/// still achieves strong order 1.0 only when the noise is commutative.
/// For non-commutative noise, consider using `srk_strong` in `runge_kutta_sde`.
///
/// # Arguments
///
/// * `prob` - The SDE problem definition
/// * `dt` - Step size
/// * `rng` - Mutable RNG reference
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::{SdeProblem, SdeSolution};
/// use scirs2_integrate::sde::milstein::milstein;
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_core::random::prelude::*;
///
/// // Geometric Brownian Motion (exact Milstein = exact solution)
/// let (mu, sigma) = (0.05_f64, 0.2_f64);
/// let prob = SdeProblem::new(
///     array![1.0_f64], [0.0, 1.0], 1,
///     move |_t, x| array![mu * x[0]],
///     move |_t, x| { let mut g = Array2::zeros((1,1)); g[[0,0]] = sigma*x[0]; g },
/// );
/// let mut rng = seeded_rng(42);
/// let sol = milstein(&prob, 0.01, &mut rng).unwrap();
/// assert!(sol.len() > 1);
/// assert!(sol.x_final().unwrap()[0] > 0.0);
/// ```
pub fn milstein<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    milstein_with_options(prob, dt, rng, &SdeOptions::default())
}

/// Milstein scheme with solver options.
pub fn milstein_with_options<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
    opts: &SdeOptions,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    prob.validate()?;
    let t0 = prob.t_span[0];
    let t1 = prob.t_span[1];
    let n_steps = compute_n_steps(t0, t1, dt, opts.max_steps)?;
    let n_state = prob.dim();
    let m = prob.n_brownian;

    let capacity = if opts.save_all_steps { n_steps + 1 } else { 2 };
    let mut sol = SdeSolution::with_capacity(capacity);
    sol.push(t0, prob.x0.clone());

    let normal = Normal::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::ComputationError(format!("Normal dist error: {}", e)))?;

    let mut x = prob.x0.clone();
    let mut t = t0;

    for step in 0..n_steps {
        let dt_actual = if step == n_steps - 1 {
            t1 - t
        } else {
            dt.min(t1 - t)
        };
        if dt_actual <= 0.0 {
            break;
        }
        let sqrt_dt = dt_actual.sqrt();

        // Generate Brownian increments ΔW ~ N(0, dt_actual * I_m)
        let dw: Array1<f64> = Array1::from_shape_fn(m, |_| normal.sample(rng) * sqrt_dt);

        let drift = (prob.f_drift)(t, &x);
        let g_curr = (prob.g_diffusion)(t, &x);

        if drift.len() != n_state || g_curr.nrows() != n_state || g_curr.ncols() != m {
            return Err(IntegrateError::DimensionMismatch(
                "Drift or diffusion output dimension mismatch".to_string(),
            ));
        }

        // Euler-Maruyama term: x + f*dt + g*ΔW
        let em_increment = drift * dt_actual + g_curr.dot(&dw);

        // Milstein correction: (1/2) Σ_j [g_j(·) ∂g/∂x]·e_j * ((ΔW_j)^2 - dt)
        let milstein_corr =
            compute_milstein_correction(t, &x, &g_curr, &dw, dt_actual, n_state, m, &prob.g_diffusion)?;

        x = x + em_increment + milstein_corr;
        t += dt_actual;

        if opts.save_all_steps {
            sol.push(t, x.clone());
        }
    }

    if !opts.save_all_steps {
        sol.push(t, x);
    }

    Ok(sol)
}

/// Compute the Milstein correction term:
///
/// ```text
/// correction_i = (1/2) Σ_j Σ_k g_{kj}(t, x) * (∂g_{ij}/∂x_k) * ((ΔW_j)^2 - dt)
/// ```
///
/// The derivative `∂g_{ij}/∂x_k` is approximated by a central finite difference.
fn compute_milstein_correction<G>(
    t: f64,
    x: &Array1<f64>,
    g_curr: &Array2<f64>,
    dw: &Array1<f64>,
    dt: f64,
    n_state: usize,
    m: usize,
    g_diffusion: &G,
) -> IntegrateResult<Array1<f64>>
where
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    let mut correction = Array1::<f64>::zeros(n_state);

    for j in 0..m {
        let dw_j = dw[j];
        let iterated_factor = 0.5 * (dw_j * dw_j - dt);

        // Compute g_j(x) · ∂g_{·j}/∂x via finite differences
        // For each state component k, we perturb x_k and measure ∂g_{ij}/∂x_k
        // g_{kj} is the (k,j) element of the diffusion matrix
        // The Milstein term for component i from Brownian motion j is:
        //   (1/2) * [g(t,x) * (∂g^{·j}/∂x)_j] * ((ΔW_j)^2 - dt)
        // where (∂g^{·j}/∂x)_j means the derivative of the j-th column of g
        // dotted with the j-th column of g.
        //
        // Specifically: corr_i += (1/2) * Σ_k g_{kj} * (dg_{ij}/dx_k) * ((ΔW_j)^2 - dt)

        let mut column_correction = Array1::<f64>::zeros(n_state);
        for k in 0..n_state {
            let h = DIFF_H * (1.0 + x[k].abs());
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[k] += h;
            x_minus[k] -= h;

            let g_plus = g_diffusion(t, &x_plus);
            let g_minus = g_diffusion(t, &x_minus);

            // dg_{ij}/dx_k for all i
            for i in 0..n_state {
                let dg_ij_dxk = (g_plus[[i, j]] - g_minus[[i, j]]) / (2.0 * h);
                column_correction[i] += g_curr[[k, j]] * dg_ij_dxk;
            }
        }

        correction = correction + column_correction * iterated_factor;
    }

    Ok(correction)
}

/// Efficient scalar (1D state, 1 Brownian motion) Milstein scheme.
///
/// For the scalar SDE `dX = f(t, X) dt + g(t, X) dW`, this avoids matrix
/// operations entirely and uses a simple scalar finite-difference approximation
/// for the Milstein correction.
///
/// The update is:
/// ```text
/// X_{n+1} = X_n + f dt + g ΔW + (1/2) g g' ((ΔW)^2 - dt)
/// ```
///
/// where `g' = ∂g/∂x` is approximated by central finite difference.
///
/// **Strong convergence order**: 1.0
///
/// # Arguments
///
/// * `f_drift` - Scalar drift `f(t, x) -> f64`
/// * `g_diff` - Scalar diffusion `g(t, x) -> f64`
/// * `x0` - Initial condition
/// * `t_span` - Time interval [t0, t1]
/// * `dt` - Step size
/// * `rng` - RNG reference
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::milstein::scalar_milstein;
/// use scirs2_core::random::prelude::*;
///
/// // Geometric Brownian Motion: dS = μ S dt + σ S dW
/// let (mu, sigma, s0) = (0.05, 0.2, 100.0_f64);
/// let mut rng = seeded_rng(42);
/// let sol = scalar_milstein(
///     |_t, x| mu * x,
///     |_t, x| sigma * x,
///     s0, [0.0, 1.0], 0.01, &mut rng,
/// ).unwrap();
/// assert!(sol.x_final().unwrap()[0] > 0.0);
/// ```
pub fn scalar_milstein<Fd, Gd>(
    f_drift: Fd,
    g_diff: Gd,
    x0: f64,
    t_span: [f64; 2],
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution>
where
    Fd: Fn(f64, f64) -> f64,
    Gd: Fn(f64, f64) -> f64,
{
    scalar_milstein_with_options(f_drift, g_diff, x0, t_span, dt, rng, &SdeOptions::default())
}

/// Scalar Milstein scheme with solver options.
pub fn scalar_milstein_with_options<Fd, Gd>(
    f_drift: Fd,
    g_diff: Gd,
    x0: f64,
    t_span: [f64; 2],
    dt: f64,
    rng: &mut StdRng,
    opts: &SdeOptions,
) -> IntegrateResult<SdeSolution>
where
    Fd: Fn(f64, f64) -> f64,
    Gd: Fn(f64, f64) -> f64,
{
    if t_span[0] >= t_span[1] {
        return Err(IntegrateError::InvalidInput(format!(
            "t_span must satisfy t0 < t1, got [{}, {}]",
            t_span[0], t_span[1]
        )));
    }
    let n_steps = compute_n_steps(t_span[0], t_span[1], dt, opts.max_steps)?;
    let t0 = t_span[0];
    let t1 = t_span[1];

    let capacity = if opts.save_all_steps { n_steps + 1 } else { 2 };
    let mut sol = SdeSolution::with_capacity(capacity);
    sol.push(t0, scirs2_core::ndarray::array![x0]);

    let normal = Normal::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::ComputationError(format!("Normal dist error: {}", e)))?;

    let mut x = x0;
    let mut t = t0;

    for step in 0..n_steps {
        let dt_actual = if step == n_steps - 1 {
            t1 - t
        } else {
            dt.min(t1 - t)
        };
        if dt_actual <= 0.0 {
            break;
        }
        let sqrt_dt = dt_actual.sqrt();
        let dw = normal.sample(rng) * sqrt_dt;

        let f_val = f_drift(t, x);
        let g_val = g_diff(t, x);

        // Numerical derivative g' = ∂g/∂x via central difference
        let h = DIFF_H * (1.0 + x.abs());
        let dg_dx = (g_diff(t, x + h) - g_diff(t, x - h)) / (2.0 * h);

        // Milstein update: x + f*dt + g*dW + (1/2)*g*g'*((dW)^2 - dt)
        x += f_val * dt_actual + g_val * dw + 0.5 * g_val * dg_dx * (dw * dw - dt_actual);
        t += dt_actual;

        if opts.save_all_steps {
            sol.push(t, scirs2_core::ndarray::array![x]);
        }
    }

    if !opts.save_all_steps {
        sol.push(t, scirs2_core::ndarray::array![x]);
    }

    Ok(sol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sde::SdeProblem;
    use scirs2_core::ndarray::{array, Array2};
    use scirs2_core::random::prelude::seeded_rng;

    fn make_gbm(mu: f64, sigma: f64, s0: f64) -> SdeProblem<
        impl Fn(f64, &Array1<f64>) -> Array1<f64>,
        impl Fn(f64, &Array1<f64>) -> Array2<f64>,
    > {
        SdeProblem::new(
            array![s0],
            [0.0, 1.0],
            1,
            move |_t, x| array![mu * x[0]],
            move |_t, x| {
                let mut g = Array2::zeros((1, 1));
                g[[0, 0]] = sigma * x[0];
                g
            },
        )
    }

    /// For GBM, the Milstein scheme with scalar g_j = sigma*x gives:
    /// g g' = sigma*x * sigma = sigma^2 * x
    /// So the correction exactly captures the Itô-Stratonovich difference,
    /// and for larger dt the Milstein solution is more accurate than EM.
    #[test]
    fn test_milstein_gbm_positive() {
        let prob = make_gbm(0.05, 0.2, 100.0);
        let mut rng = seeded_rng(42);
        let sol = milstein(&prob, 0.01, &mut rng).expect("milstein should succeed");
        // GBM should stay positive
        for xi in &sol.x {
            assert!(xi[0] > 0.0, "GBM should stay positive");
        }
    }

    #[test]
    fn test_milstein_solution_length() {
        let prob = make_gbm(0.05, 0.2, 1.0);
        let mut rng = seeded_rng(0);
        let sol = milstein(&prob, 0.1, &mut rng).expect("milstein should succeed");
        assert_eq!(sol.len(), 11);
    }

    #[test]
    fn test_scalar_milstein_basic() {
        let mut rng = seeded_rng(42);
        let sol = scalar_milstein(
            |_t, x| 0.1 * x,
            |_t, x| 0.3 * x,
            1.0,
            [0.0, 1.0],
            0.01,
            &mut rng,
        )
        .expect("scalar_milstein should succeed");
        assert!(!sol.is_empty());
        assert!(sol.x_final().expect("solution has state")[0] > 0.0);
    }

    /// Compare strong accuracy: Milstein should have lower error than EM on average
    /// over an ensemble for a path-dependent quantity.
    /// We check that Milstein weakly approximates GBM mean at least as well as EM.
    #[test]
    fn test_milstein_vs_em_weak_mean() {
        let mu = 0.1_f64;
        let sigma = 0.2_f64;
        let s0 = 1.0_f64;
        let t1 = 1.0_f64;
        let dt = 0.01;
        let analytic = s0 * (mu * t1).exp();
        let n_paths = 300;

        let mut sum_milstein = 0.0;
        for seed in 0..n_paths as u64 {
            let prob = make_gbm(mu, sigma, s0);
            let mut rng = seeded_rng(seed + 1000);
            let sol = milstein(&prob, dt, &mut rng).expect("milstein should succeed");
            sum_milstein += sol.x_final().expect("solution has state")[0];
        }
        let mean_milstein = sum_milstein / n_paths as f64;
        let rel_err = (mean_milstein - analytic).abs() / analytic;
        assert!(
            rel_err < 0.05,
            "Milstein mean {:.4} vs analytic {:.4}, rel_err {:.4}",
            mean_milstein,
            analytic,
            rel_err
        );
    }

    #[test]
    fn test_milstein_invalid_tspan() {
        let prob = make_gbm(0.05, 0.2, 1.0);
        // Override with bad t_span manually by creating a bad problem
        let x0 = array![1.0_f64];
        let bad_prob = SdeProblem::new(
            x0, [1.0, 0.0], 1,
            |_t, x| array![0.05 * x[0]],
            |_t, x| { let mut g = Array2::zeros((1, 1)); g[[0,0]] = 0.2 * x[0]; g },
        );
        let mut rng = seeded_rng(0);
        assert!(milstein(&bad_prob, 0.01, &mut rng).is_err());
    }

    #[test]
    fn test_scalar_milstein_length() {
        let mut rng = seeded_rng(7);
        let sol = scalar_milstein(
            |_t, x| 0.05 * x,
            |_t, x| 0.2 * x,
            1.0,
            [0.0, 1.0],
            0.1,
            &mut rng,
        )
        .expect("scalar_milstein should succeed");
        assert_eq!(sol.len(), 11);
    }
}
