//! Physics Simulation Datasets.
//!
//! This module provides classic physics simulation trajectories suitable for
//! testing dynamical-systems analysis, chaos detection, embedding methods,
//! and numerical-integration benchmarks.
//!
//! # Available generators
//!
//! | Function | System | Dims |
//! |---|---|---|
//! | [`lorenz_attractor`] | Lorenz (chaotic) | 3 |
//! | [`duffing_oscillator`] | Duffing (nonlinear) | 2 |
//! | [`double_pendulum`] | Double pendulum (chaotic) | 4 |
//! | [`n_body_problem`] | N-body gravitational | 6N |
//! | [`van_der_pol`] | Van der Pol (limit cycle) | 2 |

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Shared output type
// ─────────────────────────────────────────────────────────────────────────────

/// The result of a physics simulation.
///
/// `states` has shape `(n_steps, n_dims)`.
#[derive(Debug, Clone)]
pub struct PhysicsDataset {
    /// Uniformly-spaced time axis of length `n_steps`.
    pub time: Array1<f64>,
    /// State trajectory — shape `(n_steps, n_dims)`.
    pub states: Array2<f64>,
    /// Human-readable description of the system.
    pub description: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a seeded RNG.
fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Validate common parameters and return an error with the given prefix.
fn check_n_dt(prefix: &str, n: usize, dt: f64) -> Result<()> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat(format!(
            "{prefix}: n_steps must be > 0"
        )));
    }
    if dt <= 0.0 {
        return Err(DatasetsError::InvalidFormat(format!(
            "{prefix}: dt must be > 0"
        )));
    }
    Ok(())
}

/// Build a time axis: `[0, dt, 2*dt, …, (n-1)*dt]`.
fn time_axis(n: usize, dt: f64) -> Array1<f64> {
    Array1::from_vec((0..n).map(|i| i as f64 * dt).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenz attractor
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Lorenz attractor.
///
/// The classic parameter set (`sigma=10`, `rho=28`, `beta=8/3`) produces
/// the iconic butterfly chaotic attractor.
#[derive(Debug, Clone)]
pub struct LorenzConfig {
    /// Prandtl number σ.  Classic value: `10.0`.
    pub sigma: f64,
    /// Rayleigh number ρ.  Classic value: `28.0`.
    pub rho: f64,
    /// Geometry parameter β.  Classic value: `8.0/3.0`.
    pub beta: f64,
    /// Integration time step.  Default: `0.01`.
    pub dt: f64,
    /// Number of steps to record.
    pub n_steps: usize,
    /// Initial state `[x0, y0, z0]`.  Default: `[1.0, 1.0, 1.0]`.
    pub init: [f64; 3],
}

impl Default for LorenzConfig {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            dt: 0.01,
            n_steps: 5000,
            init: [1.0, 1.0, 1.0],
        }
    }
}

/// Generate a Lorenz attractor trajectory using 4th-order Runge-Kutta.
///
/// The Lorenz equations:
/// ```text
/// dx/dt = σ (y − x)
/// dy/dt = x (ρ − z) − y
/// dz/dt = x y − β z
/// ```
///
/// # Errors
///
/// Returns an error when `n_steps == 0` or `dt ≤ 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::physics::{LorenzConfig, lorenz_attractor};
///
/// let ds = lorenz_attractor(LorenzConfig::default()).expect("lorenz failed");
/// assert_eq!(ds.states.nrows(), 5000);
/// assert_eq!(ds.states.ncols(), 3);
/// ```
pub fn lorenz_attractor(config: LorenzConfig) -> Result<PhysicsDataset> {
    check_n_dt("lorenz_attractor", config.n_steps, config.dt)?;

    let LorenzConfig {
        sigma,
        rho,
        beta,
        dt,
        n_steps,
        init: [mut x, mut y, mut z],
    } = config;

    let deriv = |x: f64, y: f64, z: f64| -> (f64, f64, f64) {
        (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)
    };

    // Warm-up: 500 steps to pull the trajectory onto the attractor.
    for _ in 0..500 {
        rk4_lorenz_step(dt, &deriv, &mut x, &mut y, &mut z);
    }

    let mut states = Array2::zeros((n_steps, 3));
    for i in 0..n_steps {
        states[[i, 0]] = x;
        states[[i, 1]] = y;
        states[[i, 2]] = z;
        rk4_lorenz_step(dt, &deriv, &mut x, &mut y, &mut z);
    }

    Ok(PhysicsDataset {
        time: time_axis(n_steps, dt),
        states,
        description: format!(
            "Lorenz attractor: sigma={sigma}, rho={rho}, beta={:.4}, dt={dt}, n={n_steps}",
            beta
        ),
    })
}

#[inline]
fn rk4_lorenz_step(
    dt: f64,
    deriv: &impl Fn(f64, f64, f64) -> (f64, f64, f64),
    x: &mut f64,
    y: &mut f64,
    z: &mut f64,
) {
    let (k1x, k1y, k1z) = deriv(*x, *y, *z);
    let (k2x, k2y, k2z) = deriv(
        *x + 0.5 * dt * k1x,
        *y + 0.5 * dt * k1y,
        *z + 0.5 * dt * k1z,
    );
    let (k3x, k3y, k3z) = deriv(
        *x + 0.5 * dt * k2x,
        *y + 0.5 * dt * k2y,
        *z + 0.5 * dt * k2z,
    );
    let (k4x, k4y, k4z) = deriv(*x + dt * k3x, *y + dt * k3y, *z + dt * k3z);
    *x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    *y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
    *z += dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);
}

// ─────────────────────────────────────────────────────────────────────────────
// Duffing oscillator
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Duffing oscillator.
///
/// The forced, damped Duffing equation:
/// ```text
/// x'' + δ x' + α x + β x³ = γ cos(ω t)
/// ```
#[derive(Debug, Clone)]
pub struct DuffingConfig {
    /// Linear stiffness (default `1.0`).
    pub alpha: f64,
    /// Cubic stiffness (default `-1.0` → double-well potential).
    pub beta: f64,
    /// Damping coefficient (default `0.3`).
    pub delta: f64,
    /// Forcing amplitude (default `0.37`).
    pub gamma: f64,
    /// Forcing angular frequency (default `1.2`).
    pub omega: f64,
    /// Integration time step.
    pub dt: f64,
    /// Number of steps to record.
    pub n_steps: usize,
    /// Initial state `[x, x_dot]`.
    pub init: [f64; 2],
}

impl Default for DuffingConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: -1.0,
            delta: 0.3,
            gamma: 0.37,
            omega: 1.2,
            dt: 0.01,
            n_steps: 5000,
            init: [1.0, 0.0],
        }
    }
}

/// Generate a Duffing oscillator trajectory using 4th-order Runge-Kutta.
///
/// # Errors
///
/// Returns an error when `n_steps == 0` or `dt ≤ 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::physics::{DuffingConfig, duffing_oscillator};
///
/// let ds = duffing_oscillator(DuffingConfig::default()).expect("duffing failed");
/// assert_eq!(ds.states.nrows(), 5000);
/// assert_eq!(ds.states.ncols(), 2);
/// ```
pub fn duffing_oscillator(config: DuffingConfig) -> Result<PhysicsDataset> {
    check_n_dt("duffing_oscillator", config.n_steps, config.dt)?;

    let DuffingConfig {
        alpha,
        beta,
        delta,
        gamma,
        omega,
        dt,
        n_steps,
        init: [mut x, mut v],
    } = config;

    // Duffing: dx/dt = v,  dv/dt = -delta*v - alpha*x - beta*x^3 + gamma*cos(omega*t)
    let deriv = |x: f64, v: f64, t: f64| -> (f64, f64) {
        let x_dot = v;
        let v_dot = -delta * v - alpha * x - beta * x * x * x + gamma * (omega * t).cos();
        (x_dot, v_dot)
    };

    // Transient: discard 500 steps.
    let mut t = 0.0_f64;
    for _ in 0..500 {
        rk4_2d_step(dt, &|x, v| deriv(x, v, t), &mut x, &mut v);
        t += dt;
    }

    let mut states = Array2::zeros((n_steps, 2));
    for i in 0..n_steps {
        states[[i, 0]] = x;
        states[[i, 1]] = v;
        rk4_2d_step(dt, &|x, v| deriv(x, v, t), &mut x, &mut v);
        t += dt;
    }

    Ok(PhysicsDataset {
        time: time_axis(n_steps, dt),
        states,
        description: format!(
            "Duffing oscillator: α={alpha}, β={beta}, δ={delta}, γ={gamma}, ω={omega}, dt={dt}"
        ),
    })
}

#[inline]
fn rk4_2d_step(
    dt: f64,
    deriv: &impl Fn(f64, f64) -> (f64, f64),
    x: &mut f64,
    v: &mut f64,
) {
    let (k1x, k1v) = deriv(*x, *v);
    let (k2x, k2v) = deriv(*x + 0.5 * dt * k1x, *v + 0.5 * dt * k1v);
    let (k3x, k3v) = deriv(*x + 0.5 * dt * k2x, *v + 0.5 * dt * k2v);
    let (k4x, k4v) = deriv(*x + dt * k3x, *v + dt * k3v);
    *x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
    *v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
}

// ─────────────────────────────────────────────────────────────────────────────
// Double pendulum
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the double pendulum.
///
/// State vector: `[θ₁, ω₁, θ₂, ω₂]` (angles in radians, angular velocities).
#[derive(Debug, Clone)]
pub struct DoublePendulumConfig {
    /// Mass of the first bob (kg).
    pub m1: f64,
    /// Mass of the second bob (kg).
    pub m2: f64,
    /// Length of the first rod (m).
    pub l1: f64,
    /// Length of the second rod (m).
    pub l2: f64,
    /// Gravitational acceleration (m/s²).
    pub g: f64,
    /// Integration time step (s).
    pub dt: f64,
    /// Number of steps to record.
    pub n_steps: usize,
    /// Initial state `[θ₁, ω₁, θ₂, ω₂]`.
    pub init: [f64; 4],
}

impl Default for DoublePendulumConfig {
    fn default() -> Self {
        Self {
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            g: 9.81,
            dt: 0.01,
            n_steps: 5000,
            init: [PI / 2.0, 0.0, PI / 4.0, 0.0],
        }
    }
}

/// Generate a double-pendulum trajectory using 4th-order Runge-Kutta.
///
/// The equations of motion use the standard Lagrangian formulation.
/// The system is chaotic for large initial displacements.
///
/// # Errors
///
/// Returns an error when `n_steps == 0`, `dt ≤ 0`, or any mass/length is ≤ 0.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::physics::{DoublePendulumConfig, double_pendulum};
///
/// let ds = double_pendulum(DoublePendulumConfig::default()).expect("pendulum failed");
/// assert_eq!(ds.states.ncols(), 4);
/// ```
pub fn double_pendulum(config: DoublePendulumConfig) -> Result<PhysicsDataset> {
    check_n_dt("double_pendulum", config.n_steps, config.dt)?;
    for (name, val) in [
        ("m1", config.m1),
        ("m2", config.m2),
        ("l1", config.l1),
        ("l2", config.l2),
        ("g", config.g),
    ] {
        if val <= 0.0 {
            return Err(DatasetsError::InvalidFormat(format!(
                "double_pendulum: {name} must be > 0"
            )));
        }
    }

    let DoublePendulumConfig {
        m1,
        m2,
        l1,
        l2,
        g,
        dt,
        n_steps,
        init,
    } = config;

    let [mut th1, mut om1, mut th2, mut om2] = init;

    // Equations of motion via the Lagrangian formulation.
    // Returns (dth1, dom1, dth2, dom2).
    let deriv = |th1: f64, om1: f64, th2: f64, om2: f64| -> (f64, f64, f64, f64) {
        let d = th1 - th2;
        let cos_d = d.cos();
        let sin_d = d.sin();
        let denom1 = (m1 + m2) * l1 - m2 * l1 * cos_d * cos_d;
        let denom2 = (l2 / l1) * denom1;

        let dom1_num = m2 * l1 * om1 * om1 * sin_d * cos_d
            + m2 * g * th2.sin() * cos_d
            + m2 * l2 * om2 * om2 * sin_d
            - (m1 + m2) * g * th1.sin();
        let dom2_num = -m2 * l2 * om2 * om2 * sin_d * cos_d
            + (m1 + m2) * g * th1.sin() * cos_d
            - (m1 + m2) * l1 * om1 * om1 * sin_d
            - (m1 + m2) * g * th2.sin();

        (
            om1,
            dom1_num / denom1,
            om2,
            dom2_num / denom2,
        )
    };

    let mut states = Array2::zeros((n_steps, 4));
    for i in 0..n_steps {
        states[[i, 0]] = th1;
        states[[i, 1]] = om1;
        states[[i, 2]] = th2;
        states[[i, 3]] = om2;

        let (k1a, k1b, k1c, k1d) = deriv(th1, om1, th2, om2);
        let (k2a, k2b, k2c, k2d) = deriv(
            th1 + 0.5 * dt * k1a,
            om1 + 0.5 * dt * k1b,
            th2 + 0.5 * dt * k1c,
            om2 + 0.5 * dt * k1d,
        );
        let (k3a, k3b, k3c, k3d) = deriv(
            th1 + 0.5 * dt * k2a,
            om1 + 0.5 * dt * k2b,
            th2 + 0.5 * dt * k2c,
            om2 + 0.5 * dt * k2d,
        );
        let (k4a, k4b, k4c, k4d) = deriv(
            th1 + dt * k3a,
            om1 + dt * k3b,
            th2 + dt * k3c,
            om2 + dt * k3d,
        );
        th1 += dt / 6.0 * (k1a + 2.0 * k2a + 2.0 * k3a + k4a);
        om1 += dt / 6.0 * (k1b + 2.0 * k2b + 2.0 * k3b + k4b);
        th2 += dt / 6.0 * (k1c + 2.0 * k2c + 2.0 * k3c + k4c);
        om2 += dt / 6.0 * (k1d + 2.0 * k2d + 2.0 * k3d + k4d);
    }

    Ok(PhysicsDataset {
        time: time_axis(n_steps, dt),
        states,
        description: format!(
            "Double pendulum: m1={m1}, m2={m2}, l1={l1}, l2={l2}, g={g}, dt={dt}"
        ),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// N-body gravitational simulation
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an N-body gravitational simulation.
///
/// Bodies are placed randomly in the unit cube with unit masses.
/// The state dimension is `6 * n_bodies` (3 positions + 3 velocities per body).
#[derive(Debug, Clone)]
pub struct NBodyConfig {
    /// Number of gravitating bodies.
    pub n_bodies: usize,
    /// Integration time step.
    pub dt: f64,
    /// Number of steps to record.
    pub n_steps: usize,
    /// Random seed for initial conditions.
    pub seed: u64,
    /// Gravitational constant G (default `1.0` in natural units).
    pub g: f64,
    /// Softening length to prevent singularities (default `0.01`).
    pub softening: f64,
}

impl Default for NBodyConfig {
    fn default() -> Self {
        Self {
            n_bodies: 3,
            dt: 0.01,
            n_steps: 1000,
            seed: 42,
            g: 1.0,
            softening: 0.01,
        }
    }
}

/// Generate an N-body gravitational simulation using 4th-order Runge-Kutta.
///
/// The state array has shape `(n_steps, 6 * n_bodies)` where each block of 6
/// columns per body is `[x, y, z, vx, vy, vz]`.
///
/// A softening length `ε` is added to prevent coordinate singularities:
/// `F = G m_i m_j / (r² + ε²)^(3/2) * r_vec`.
///
/// # Errors
///
/// Returns an error when `n_bodies < 2`, `n_steps == 0`, or `dt ≤ 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::physics::{NBodyConfig, n_body_problem};
///
/// let mut cfg = NBodyConfig::default();
/// cfg.n_steps = 200;
/// let ds = n_body_problem(cfg).expect("n-body failed");
/// assert_eq!(ds.states.ncols(), 18); // 3 bodies × 6 dims
/// ```
pub fn n_body_problem(config: NBodyConfig) -> Result<PhysicsDataset> {
    check_n_dt("n_body_problem", config.n_steps, config.dt)?;
    if config.n_bodies < 2 {
        return Err(DatasetsError::InvalidFormat(
            "n_body_problem: n_bodies must be >= 2".to_string(),
        ));
    }

    let NBodyConfig {
        n_bodies,
        dt,
        n_steps,
        seed,
        g,
        softening,
    } = config;

    let n_state = 6 * n_bodies;
    let eps2 = softening * softening;

    // Initialize positions and velocities randomly in [-1, 1].
    let mut rng = make_rng(seed);
    let uniform = scirs2_core::random::Uniform::new(-1.0_f64, 1.0_f64).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform distribution creation failed: {e}"))
    })?;
    let masses: Vec<f64> = (0..n_bodies).map(|_| 1.0_f64).collect();

    let mut state: Vec<f64> = (0..n_state).map(|_| uniform.sample(&mut rng)).collect();

    // Centre-of-mass correction so the system does not drift.
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    let mut vx_cm = 0.0_f64;
    let mut vy_cm = 0.0_f64;
    let mut vz_cm = 0.0_f64;
    let total_mass: f64 = masses.iter().sum();
    for i in 0..n_bodies {
        let b = 6 * i;
        cx += state[b];
        cy += state[b + 1];
        cz += state[b + 2];
        vx_cm += state[b + 3];
        vy_cm += state[b + 4];
        vz_cm += state[b + 5];
    }
    cx /= total_mass;
    cy /= total_mass;
    cz /= total_mass;
    vx_cm /= total_mass;
    vy_cm /= total_mass;
    vz_cm /= total_mass;
    for i in 0..n_bodies {
        let b = 6 * i;
        state[b] -= cx;
        state[b + 1] -= cy;
        state[b + 2] -= cz;
        state[b + 3] -= vx_cm;
        state[b + 4] -= vy_cm;
        state[b + 5] -= vz_cm;
    }

    // Derivative function.
    let compute_deriv = |s: &[f64]| -> Vec<f64> {
        let mut ds = vec![0.0_f64; n_state];
        for i in 0..n_bodies {
            let bi = 6 * i;
            // velocity → position derivative
            ds[bi] = s[bi + 3];
            ds[bi + 1] = s[bi + 4];
            ds[bi + 2] = s[bi + 5];
            // gravitational acceleration from all other bodies
            let mut ax = 0.0_f64;
            let mut ay = 0.0_f64;
            let mut az = 0.0_f64;
            for j in 0..n_bodies {
                if i == j {
                    continue;
                }
                let bj = 6 * j;
                let dx = s[bj] - s[bi];
                let dy = s[bj + 1] - s[bi + 1];
                let dz = s[bj + 2] - s[bi + 2];
                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                let r3_inv = 1.0 / (r2.sqrt() * r2);
                let f = g * masses[j] * r3_inv;
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }
            ds[bi + 3] = ax;
            ds[bi + 4] = ay;
            ds[bi + 5] = az;
        }
        ds
    };

    // RK4 step for arbitrary dimension.
    let rk4_step = |s: &mut Vec<f64>, dt: f64| {
        let k1 = compute_deriv(s);
        let s2: Vec<f64> = s.iter().zip(k1.iter()).map(|(x, k)| x + 0.5 * dt * k).collect();
        let k2 = compute_deriv(&s2);
        let s3: Vec<f64> = s.iter().zip(k2.iter()).map(|(x, k)| x + 0.5 * dt * k).collect();
        let k3 = compute_deriv(&s3);
        let s4: Vec<f64> = s.iter().zip(k3.iter()).map(|(x, k)| x + dt * k).collect();
        let k4 = compute_deriv(&s4);
        for idx in 0..s.len() {
            s[idx] += dt / 6.0 * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]);
        }
    };

    let mut out = Array2::zeros((n_steps, n_state));
    for i in 0..n_steps {
        for j in 0..n_state {
            out[[i, j]] = state[j];
        }
        rk4_step(&mut state, dt);
    }

    Ok(PhysicsDataset {
        time: time_axis(n_steps, dt),
        states: out,
        description: format!(
            "N-body problem: n_bodies={n_bodies}, G={g}, softening={softening}, dt={dt}"
        ),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Van der Pol oscillator
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Van der Pol oscillator trajectory using 4th-order Runge-Kutta.
///
/// The Van der Pol equation:
/// ```text
/// x'' − μ (1 − x²) x' + x = 0
/// ```
/// which in state-space form becomes:
/// ```text
/// dx/dt = y
/// dy/dt = μ (1 − x²) y − x
/// ```
///
/// For `μ = 0` the system reduces to a simple harmonic oscillator.
/// For large `μ` it exhibits a strongly nonlinear limit cycle with relaxation
/// oscillations.
///
/// # Arguments
///
/// * `mu`      – Nonlinearity parameter (must be ≥ 0).
/// * `dt`      – Integration step size (must be > 0).
/// * `n_steps` – Number of steps to record (must be > 0).
///
/// # Errors
///
/// Returns an error when `mu < 0`, `dt ≤ 0`, or `n_steps == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::physics::van_der_pol;
///
/// let ds = van_der_pol(1.0, 0.01, 2000).expect("vdp failed");
/// assert_eq!(ds.states.nrows(), 2000);
/// assert_eq!(ds.states.ncols(), 2);
/// ```
pub fn van_der_pol(mu: f64, dt: f64, n_steps: usize) -> Result<PhysicsDataset> {
    check_n_dt("van_der_pol", n_steps, dt)?;
    if mu < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "van_der_pol: mu must be >= 0".to_string(),
        ));
    }

    let deriv = |x: f64, y: f64| -> (f64, f64) { (y, mu * (1.0 - x * x) * y - x) };

    let mut x = 2.0_f64;
    let mut y = 0.0_f64;

    // Warm-up to attract to limit cycle.
    for _ in 0..500 {
        rk4_2d_step(dt, &deriv, &mut x, &mut y);
    }

    let mut states = Array2::zeros((n_steps, 2));
    for i in 0..n_steps {
        states[[i, 0]] = x;
        states[[i, 1]] = y;
        rk4_2d_step(dt, &deriv, &mut x, &mut y);
    }

    Ok(PhysicsDataset {
        time: time_axis(n_steps, dt),
        states,
        description: format!("Van der Pol oscillator: mu={mu}, dt={dt}, n={n_steps}"),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Lorenz ──────────────────────────────────────────────────────────────

    #[test]
    fn test_lorenz_shape() {
        let cfg = LorenzConfig {
            n_steps: 1000,
            ..Default::default()
        };
        let ds = lorenz_attractor(cfg).expect("lorenz failed");
        assert_eq!(ds.states.nrows(), 1000);
        assert_eq!(ds.states.ncols(), 3);
        assert_eq!(ds.time.len(), 1000);
    }

    #[test]
    fn test_lorenz_time_axis() {
        let cfg = LorenzConfig {
            n_steps: 100,
            dt: 0.05,
            ..Default::default()
        };
        let ds = lorenz_attractor(cfg).expect("lorenz failed");
        assert!((ds.time[0] - 0.0).abs() < 1e-12);
        assert!((ds.time[99] - 99.0 * 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_lorenz_chaotic_divergence() {
        // Two trajectories separated by a tiny perturbation must diverge
        // significantly over time (positive Lyapunov exponent).
        let make = |eps: f64| {
            lorenz_attractor(LorenzConfig {
                n_steps: 2000,
                dt: 0.01,
                init: [1.0 + eps, 1.0, 1.0],
                ..Default::default()
            })
            .expect("lorenz failed")
        };
        let ds1 = make(0.0);
        let ds2 = make(1e-8);
        let n = 2000;
        let dx = ds1.states[[n - 1, 0]] - ds2.states[[n - 1, 0]];
        let dy = ds1.states[[n - 1, 1]] - ds2.states[[n - 1, 1]];
        let dz = ds1.states[[n - 1, 2]] - ds2.states[[n - 1, 2]];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        // Should grow much larger than the initial 1e-8 separation.
        assert!(dist > 1e-4, "Expected divergence, got dist={dist}");
    }

    #[test]
    fn test_lorenz_error_zero_steps() {
        let cfg = LorenzConfig {
            n_steps: 0,
            ..Default::default()
        };
        assert!(lorenz_attractor(cfg).is_err());
    }

    #[test]
    fn test_lorenz_error_bad_dt() {
        let cfg = LorenzConfig {
            dt: -0.01,
            ..Default::default()
        };
        assert!(lorenz_attractor(cfg).is_err());
    }

    // ── Duffing ─────────────────────────────────────────────────────────────

    #[test]
    fn test_duffing_shape() {
        let cfg = DuffingConfig {
            n_steps: 800,
            ..Default::default()
        };
        let ds = duffing_oscillator(cfg).expect("duffing failed");
        assert_eq!(ds.states.nrows(), 800);
        assert_eq!(ds.states.ncols(), 2);
    }

    #[test]
    fn test_duffing_error_bad_dt() {
        let cfg = DuffingConfig {
            dt: 0.0,
            ..Default::default()
        };
        assert!(duffing_oscillator(cfg).is_err());
    }

    // ── Double pendulum ──────────────────────────────────────────────────────

    #[test]
    fn test_double_pendulum_shape() {
        let cfg = DoublePendulumConfig {
            n_steps: 500,
            ..Default::default()
        };
        let ds = double_pendulum(cfg).expect("pendulum failed");
        assert_eq!(ds.states.nrows(), 500);
        assert_eq!(ds.states.ncols(), 4);
    }

    #[test]
    fn test_double_pendulum_energy_approximately_conserved() {
        // For a small-angle initial condition (far from chaos) and fine time step,
        // total mechanical energy should be conserved to within ~1%.
        let cfg = DoublePendulumConfig {
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            g: 9.81,
            dt: 0.001,
            n_steps: 2000,
            init: [0.1, 0.0, 0.1, 0.0], // small-angle regime
        };
        let ds = double_pendulum(cfg.clone()).expect("pendulum failed");
        let m1 = cfg.m1;
        let m2 = cfg.m2;
        let l1 = cfg.l1;
        let l2 = cfg.l2;
        let g = cfg.g;

        let energy = |row: usize| -> f64 {
            let th1 = ds.states[[row, 0]];
            let om1 = ds.states[[row, 1]];
            let th2 = ds.states[[row, 2]];
            let om2 = ds.states[[row, 3]];
            // Kinetic energy
            let ke = 0.5 * (m1 + m2) * l1 * l1 * om1 * om1
                + 0.5 * m2 * l2 * l2 * om2 * om2
                + m2 * l1 * l2 * om1 * om2 * (th1 - th2).cos();
            // Potential energy (reference: top pivot)
            let pe = -(m1 + m2) * g * l1 * th1.cos() - m2 * g * l2 * th2.cos();
            ke + pe
        };

        let e0 = energy(0);
        let e_final = energy(ds.states.nrows() - 1);
        let rel_err = ((e_final - e0) / e0.abs().max(1e-12)).abs();
        assert!(rel_err < 0.02, "Energy drift too large: {rel_err:.4}");
    }

    #[test]
    fn test_double_pendulum_error_zero_mass() {
        let cfg = DoublePendulumConfig {
            m1: 0.0,
            ..Default::default()
        };
        assert!(double_pendulum(cfg).is_err());
    }

    // ── N-body ───────────────────────────────────────────────────────────────

    #[test]
    fn test_nbody_shape() {
        let cfg = NBodyConfig {
            n_bodies: 4,
            n_steps: 200,
            ..Default::default()
        };
        let ds = n_body_problem(cfg).expect("n-body failed");
        assert_eq!(ds.states.nrows(), 200);
        assert_eq!(ds.states.ncols(), 24); // 4 × 6
    }

    #[test]
    fn test_nbody_error_too_few_bodies() {
        let cfg = NBodyConfig {
            n_bodies: 1,
            ..Default::default()
        };
        assert!(n_body_problem(cfg).is_err());
    }

    // ── Van der Pol ──────────────────────────────────────────────────────────

    #[test]
    fn test_van_der_pol_shape() {
        let ds = van_der_pol(1.5, 0.01, 1000).expect("vdp failed");
        assert_eq!(ds.states.nrows(), 1000);
        assert_eq!(ds.states.ncols(), 2);
    }

    #[test]
    fn test_van_der_pol_limit_cycle_amplitude() {
        // For mu > 0 the amplitude should be close to 2 after warm-up.
        let ds = van_der_pol(1.0, 0.01, 3000).expect("vdp failed");
        let max_x = (0..ds.states.nrows())
            .map(|i| ds.states[[i, 0]].abs())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_x > 1.5 && max_x < 2.5, "unexpected amplitude: {max_x}");
    }

    #[test]
    fn test_van_der_pol_error_negative_mu() {
        assert!(van_der_pol(-0.1, 0.01, 100).is_err());
    }

    #[test]
    fn test_van_der_pol_harmonic_oscillator_period() {
        // For mu=0, Van der Pol becomes a harmonic oscillator with period 2π.
        // Check that the trajectory crosses zero with approximately period 2π.
        let dt = 0.001;
        let n = 20_000;
        let ds = van_der_pol(0.0, dt, n).expect("vdp failed");
        // Count upward zero-crossings of x.
        let mut crossings = vec![];
        for i in 1..n {
            let x_prev = ds.states[[i - 1, 0]];
            let x_curr = ds.states[[i, 0]];
            if x_prev < 0.0 && x_curr >= 0.0 {
                crossings.push(i as f64 * dt);
            }
        }
        if crossings.len() >= 2 {
            let period = crossings[crossings.len() - 1] - crossings[crossings.len() - 2];
            let expected = 2.0 * PI;
            let rel = (period - expected).abs() / expected;
            assert!(rel < 0.02, "period={period:.4}, expected={expected:.4}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Submodules (added in v0.3.0)
// ─────────────────────────────────────────────────────────────────────────────

/// Standard ODE system datasets for testing integrators.
pub mod ode_systems;

/// Chaotic map and attractor datasets.
pub mod chaotic_systems;

/// Fluid dynamics analytical and numerical datasets.
pub mod fluid;
