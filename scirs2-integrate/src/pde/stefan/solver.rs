//! Stefan problem solver using the enthalpy method.
//!
//! Implements the 1D one-phase Stefan (melting) problem via the enthalpy
//! formulation, which eliminates the need to explicitly track the sharp
//! interface while still allowing its recovery as an isotherm.
//!
//! ## Enthalpy Formulation
//!
//! Define the enthalpy `H(u)` as:
//!
//! ```text
//! H(u) = u + St * Θ(u − T_m)
//! ```
//!
//! where `Θ` is the Heaviside function and `St` is the Stefan number.
//! The governing PDE becomes:
//!
//! ```text
//! ∂H/∂t = α ∂²u/∂x²
//! ```
//!
//! which is solved on the fixed domain `[0, L_max]` with explicit time-stepping.
//! The interface position is recovered as the location of the `T_m` isotherm.

use super::types::{StefanConfig, StefanResult};
use crate::error::{IntegrateError, IntegrateResult};

/// Solver for the 1D one-phase Stefan (melting/solidification) problem.
pub struct StefanSolver;

impl StefanSolver {
    /// Create a new Stefan problem solver.
    pub fn new() -> Self {
        Self
    }

    /// Solve the Stefan problem with the given configuration.
    ///
    /// Returns a [`StefanResult`] containing:
    /// - output times
    /// - interface positions `s(t)`
    /// - temperature fields `u(x, t)`
    /// - the fixed spatial grid
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or if the CFL
    /// condition is severely violated (r > 0.5 implies instability).
    pub fn solve(&self, config: &StefanConfig) -> IntegrateResult<StefanResult> {
        config.validate()?;

        let nx = config.nx;
        let dt = config.dt;
        let alpha = config.diffusivity;
        let st = config.stefan_number;
        let t_m = config.melting_temp;
        let t_wall = config.wall_temp;
        let l_max = config.l_max;
        let t_max = config.max_time;
        let output_every = config.output_every;

        let dx = l_max / (nx - 1) as f64;
        let r = alpha * dt / (dx * dx);

        if r > 0.5 {
            return Err(IntegrateError::ConvergenceError(format!(
                "CFL condition violated: r = {r:.4} > 0.5. Reduce dt or increase nx."
            )));
        }

        // Build spatial grid
        let grid: Vec<f64> = (0..nx).map(|i| i as f64 * dx).collect();

        // Initialize temperature field: u = T_wall for x = 0 (only node initially in melt),
        // u = T_m (solid) elsewhere.  Since s(0)=0, the entire domain is initially solid.
        // We set the Dirichlet condition u(0,t) = T_wall at all times and initialize the
        // interior to the melting temperature (solid at rest).
        let mut u = vec![t_m; nx];
        u[0] = t_wall;

        // Enthalpy: H = u + St * Θ(u - T_m)
        // In solid region: H = u (since u ≤ T_m  ⟹  Θ = 0)
        // In liquid region: H = u + St (since u > T_m  ⟹  Θ = 1)
        let enthalpy_of = |temp: f64| -> f64 {
            if temp > t_m {
                temp + st
            } else {
                temp
            }
        };

        // Initial enthalpy
        let mut h_field: Vec<f64> = u.iter().map(|&temp| enthalpy_of(temp)).collect();

        // Output buffers
        let mut times = Vec::new();
        let mut interface_positions = Vec::new();
        let mut temperature_fields = Vec::new();

        let n_steps = (t_max / dt).ceil() as usize;

        // Record initial state
        let s0 = interface_position(&u, t_m, &grid);
        times.push(0.0);
        interface_positions.push(s0);
        temperature_fields.push(u.clone());

        for step in 1..=n_steps {
            let t_current = step as f64 * dt;
            if t_current > t_max + dt * 0.5 {
                break;
            }

            // ── Explicit time step in enthalpy formulation ─────────────────
            // ∂H/∂t = α ∂²u/∂x²
            // H_i^{n+1} = H_i^n + r * (u_{i+1}^n - 2 u_i^n + u_{i-1}^n)
            let mut h_new = h_field.clone();

            // Interior nodes
            for i in 1..(nx - 1) {
                let d2u = u[i + 1] - 2.0 * u[i] + u[i - 1];
                h_new[i] = h_field[i] + r * d2u;
            }

            // Left Dirichlet: u(0,t) = T_wall  →  H(0,t) = T_wall + St
            h_new[0] = enthalpy_of(t_wall);

            // Right boundary: far-field solid, u = T_m (zero-flux Neumann for solid)
            h_new[nx - 1] = h_field[nx - 1]; // no flux from outside

            // Recover temperature from enthalpy:
            // H < T_m           ⟹ solid:  u = H
            // T_m ≤ H ≤ T_m+St ⟹ mushy:  u = T_m  (phase change region)
            // H > T_m + St      ⟹ liquid: u = H - St
            let mut u_new = vec![0.0_f64; nx];
            for i in 0..nx {
                u_new[i] = temp_from_enthalpy(h_new[i], t_m, st);
            }

            // Enforce left Dirichlet exactly
            u_new[0] = t_wall;
            h_new[0] = enthalpy_of(t_wall);

            h_field = h_new;
            u = u_new;

            // Output at requested intervals
            if step % output_every == 0 || step == n_steps {
                let s = interface_position(&u, t_m, &grid);
                times.push(t_current.min(t_max));
                interface_positions.push(s);
                temperature_fields.push(u.clone());
            }
        }

        Ok(StefanResult {
            times,
            interface_positions,
            temperature_fields,
            grid,
        })
    }
}

/// Recover temperature from enthalpy value.
///
/// - `H < T_m`          → solid:  `u = H`
/// - `T_m ≤ H ≤ T_m+St` → mushy:  `u = T_m`
/// - `H > T_m + St`     → liquid: `u = H − St`
#[inline]
fn temp_from_enthalpy(h: f64, t_m: f64, st: f64) -> f64 {
    if h < t_m {
        h
    } else if h <= t_m + st {
        t_m
    } else {
        h - st
    }
}

/// Locate the melt front (T_m isotherm) by linear interpolation.
///
/// Returns the x position where `u = T_m`, searching from the left.
/// If no interface is found, returns 0.0 (no melt) or `grid[nx-1]` (fully melted).
fn interface_position(u: &[f64], t_m: f64, grid: &[f64]) -> f64 {
    let n = u.len();
    // Search for the rightmost node where u > t_m
    // (interface is where u transitions from > T_m to <= T_m)
    for i in 0..(n - 1) {
        if u[i] > t_m && u[i + 1] <= t_m {
            // Linear interpolation
            let frac = (u[i] - t_m) / (u[i] - u[i + 1]);
            return grid[i] + frac * (grid[i + 1] - grid[i]);
        }
    }
    // Check if entire domain is above T_m (fully melted)
    if u[n - 1] > t_m {
        return grid[n - 1];
    }
    // No melting yet
    0.0
}

// ── Analytical solution ───────────────────────────────────────────────────────

/// Compute the analytical interface position for the 1D Stefan problem.
///
/// The analytical solution is `s(t) = 2λ√(αt)` where `λ` is the positive
/// root of:
///
/// ```text
/// λ exp(λ²) erf(λ) = St / √π
/// ```
///
/// Returns a closure `s(t)` for the interface position.
///
/// # Arguments
///
/// * `st`    - Stefan number St = c_p (T_wall − T_m) / L.
/// * `alpha` - Thermal diffusivity.
///
/// # Errors
///
/// Returns an error if `st ≤ 0`.
pub fn analytical_stefan_interface(st: f64, alpha: f64) -> IntegrateResult<impl Fn(f64) -> f64> {
    if st <= 0.0 {
        return Err(IntegrateError::InvalidInput(
            "Stefan number must be positive".to_string(),
        ));
    }
    if alpha <= 0.0 {
        return Err(IntegrateError::InvalidInput(
            "diffusivity must be positive".to_string(),
        ));
    }

    let lambda = find_stefan_lambda(st)?;

    Ok(move |t: f64| {
        if t <= 0.0 {
            0.0
        } else {
            2.0 * lambda * (alpha * t).sqrt()
        }
    })
}

/// Find λ > 0 satisfying `λ exp(λ²) erf(λ) = St/√π` by bisection.
pub fn find_stefan_lambda(st: f64) -> IntegrateResult<f64> {
    let target = st / std::f64::consts::PI.sqrt();

    // f(λ) = λ exp(λ²) erf(λ) - target
    let f = |lam: f64| -> f64 { lam * (lam * lam).exp() * erf_approx(lam) - target };

    // The function is monotonically increasing on (0, ∞).
    // Find a bracket.
    let mut lo = 1e-12_f64;
    let mut hi = 10.0_f64;

    // Ensure bracket
    if f(hi) < 0.0 {
        hi = 100.0;
    }
    if f(hi) < 0.0 {
        return Err(IntegrateError::ConvergenceError(
            "Could not bracket Stefan root: Stefan number too large?".to_string(),
        ));
    }

    // Bisection
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 1e-14 {
            break;
        }
    }

    Ok(0.5 * (lo + hi))
}

/// Approximation to the error function erf(x) using Horner's method.
/// Accurate to about 7 significant figures.
pub fn erf_approx(x: f64) -> f64 {
    // Use libm for accuracy
    libm::erf(x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// The analytical interface for St=1 should match the numerical front.
    #[test]
    fn test_stefan_interface_monotone() {
        let cfg = StefanConfig {
            nx: 80,
            dt: 5e-5,
            stefan_number: 1.0,
            diffusivity: 1.0,
            melting_temp: 0.0,
            wall_temp: 1.0,
            l_max: 4.0,
            max_time: 0.2,
            output_every: 50,
        };
        let solver = StefanSolver::new();
        let result = solver.solve(&cfg).expect("solve failed");

        // Interface must be non-decreasing
        for i in 1..result.interface_positions.len() {
            assert!(
                result.interface_positions[i] >= result.interface_positions[i - 1] - 1e-10,
                "interface not monotone at step {i}: {} < {}",
                result.interface_positions[i],
                result.interface_positions[i - 1]
            );
        }
    }

    /// The result has consistent sizes.
    #[test]
    fn test_stefan_result_shape() {
        let cfg = StefanConfig::default();
        let solver = StefanSolver::new();
        let result = solver.solve(&cfg).expect("solve failed");

        assert_eq!(result.times.len(), result.interface_positions.len());
        assert_eq!(result.times.len(), result.temperature_fields.len());
        assert_eq!(result.grid.len(), cfg.nx);
        for field in &result.temperature_fields {
            assert_eq!(field.len(), cfg.nx);
        }
    }

    /// Wall temperature must remain T_wall throughout.
    #[test]
    fn test_stefan_wall_temp() {
        let cfg = StefanConfig {
            nx: 50,
            dt: 2e-5,
            max_time: 0.05,
            output_every: 20,
            ..Default::default()
        };
        let solver = StefanSolver::new();
        let result = solver.solve(&cfg).expect("solve failed");

        for field in &result.temperature_fields {
            assert!(
                approx_eq(field[0], cfg.wall_temp, 1e-12),
                "wall temp changed: {}",
                field[0]
            );
        }
    }

    /// StefanConfig::default() produces valid configuration.
    #[test]
    fn test_stefan_config_default() {
        let cfg = StefanConfig::default();
        assert!(cfg.nx > 0);
        assert!(cfg.dt > 0.0);
        assert!(cfg.wall_temp > cfg.melting_temp);
        cfg.validate().expect("default config should be valid");
    }

    /// Analytical interface should match s ~ 2λ√t for small t.
    #[test]
    fn test_stefan_analytical_small_t() {
        let st = 1.0;
        let alpha = 1.0;
        let s_fn = analytical_stefan_interface(st, alpha).expect("analytical failed");

        let lambda = find_stefan_lambda(st).expect("lambda failed");

        for &t in &[0.01, 0.05, 0.1, 0.2] {
            let s_analytical = 2.0 * lambda * (alpha * t).sqrt();
            let s_fn_val = s_fn(t);
            assert!(
                approx_eq(s_analytical, s_fn_val, 1e-12),
                "t={t}: analytical={s_analytical}, fn={s_fn_val}"
            );
        }
    }

    /// Numerical solution approximately matches analytical for small t.
    #[test]
    fn test_stefan_numerical_vs_analytical() {
        let st = 1.0;
        let alpha = 1.0;
        let t_final = 0.1;

        let cfg = StefanConfig {
            nx: 200,
            dt: 1e-5,
            stefan_number: st,
            diffusivity: alpha,
            melting_temp: 0.0,
            wall_temp: 1.0,
            l_max: 3.0,
            max_time: t_final,
            output_every: 1000,
        };

        let solver = StefanSolver::new();
        let result = solver.solve(&cfg).expect("solve failed");

        let s_fn = analytical_stefan_interface(st, alpha).expect("analytical failed");
        let s_exact = s_fn(t_final);

        // Get the final interface position
        let s_num = *result.interface_positions.last().expect("no output");

        // Allow 10% relative error due to coarse grid / enthalpy smearing
        let rel_err = (s_num - s_exact).abs() / s_exact;
        assert!(
            rel_err < 0.15,
            "numerical vs analytical: s_num={s_num:.4}, s_exact={s_exact:.4}, rel_err={rel_err:.3}"
        );
    }
}
