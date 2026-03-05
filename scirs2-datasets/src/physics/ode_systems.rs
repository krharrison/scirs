//! Standard ODE system dataset generators for testing integrators.
//!
//! All integrations use the classical 4th-order Runge-Kutta method (RK4)
//! unless stated otherwise.  The integrators return a pair `(t, states)` where
//! `t` is a uniformly-spaced time axis and `states[i]` is the system state at
//! time `t[i]`.
//!
//! # Systems implemented
//!
//! | Function | Description | Dim |
//! |---|---|---|
//! | [`van_der_pol_ode`] | Van der Pol oscillator | 2 |
//! | [`lotka_volterra`] | Predator-prey system | 2 |
//! | [`lorenz63`] | Lorenz attractor (RK4) | 3 |
//! | [`roessler`] | Rössler attractor | 3 |
//! | [`duffing_ode`] | Forced Duffing oscillator | 2 |
//! | [`pendulum`] | Simple pendulum | 2 |
//! | [`double_pendulum_ode`] | Double pendulum (Lagrangian) | 4 |

use crate::error::{DatasetsError, Result};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper: classical 4th-order Runge-Kutta
// ─────────────────────────────────────────────────────────────────────────────

/// Integrate a generic first-order ODE `dy/dt = f(t, y)` over a uniform grid.
///
/// Returns `(t_vec, states)` where each element of `states` is a snapshot
/// `y(t_i)` for `i = 0 … n_steps-1`.
fn rk4<const N: usize, F>(
    f: F,
    t_span: (f64, f64),
    dt: f64,
    y0: [f64; N],
) -> Result<(Vec<f64>, Vec<[f64; N]>)>
where
    F: Fn(f64, &[f64; N]) -> [f64; N],
{
    let (t0, t1) = t_span;
    if dt <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "rk4: dt must be positive".into(),
        ));
    }
    if t1 <= t0 {
        return Err(DatasetsError::InvalidFormat(
            "rk4: t_span end must be > start".into(),
        ));
    }

    let n_steps = ((t1 - t0) / dt).ceil() as usize + 1;
    let mut t_vec = Vec::with_capacity(n_steps);
    let mut states = Vec::with_capacity(n_steps);

    let mut t = t0;
    let mut y = y0;

    t_vec.push(t);
    states.push(y);

    while t < t1 {
        let actual_dt = if t + dt > t1 { t1 - t } else { dt };
        let k1 = f(t, &y);
        let mut tmp = [0.0f64; N];
        for i in 0..N {
            tmp[i] = y[i] + 0.5 * actual_dt * k1[i];
        }
        let k2 = f(t + 0.5 * actual_dt, &tmp);
        for i in 0..N {
            tmp[i] = y[i] + 0.5 * actual_dt * k2[i];
        }
        let k3 = f(t + 0.5 * actual_dt, &tmp);
        for i in 0..N {
            tmp[i] = y[i] + actual_dt * k3[i];
        }
        let k4 = f(t + actual_dt, &tmp);

        for i in 0..N {
            y[i] += actual_dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += actual_dt;
        t_vec.push(t);
        states.push(y);
    }
    Ok((t_vec, states))
}

// ─────────────────────────────────────────────────────────────────────────────
// Van der Pol oscillator
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Van der Pol oscillator trajectory.
///
/// The equations are:
/// ```text
/// ẏ₁ = y₂
/// ẏ₂ = μ·(1 - y₁²)·y₂ - y₁
/// ```
///
/// # Parameters
/// - `mu`     — nonlinearity parameter (must be ≥ 0)
/// - `t_span` — `(t_start, t_end)`
/// - `dt`     — integration step size
/// - `y0`     — initial state `[y₁₀, y₂₀]`
///
/// # Returns
/// `(t_vec, states)` where each element of `states` is `[y₁, y₂]`.
pub fn van_der_pol_ode(
    mu: f64,
    t_span: (f64, f64),
    dt: f64,
    y0: [f64; 2],
) -> Result<(Vec<f64>, Vec<[f64; 2]>)> {
    if mu < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "van_der_pol_ode: mu must be >= 0".into(),
        ));
    }
    rk4(
        |_t, y| {
            let dy1 = y[1];
            let dy2 = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
            [dy1, dy2]
        },
        t_span,
        dt,
        y0,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Lotka-Volterra (predator-prey)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Lotka-Volterra predator-prey trajectory.
///
/// The equations are:
/// ```text
/// ẋ = α·x - β·x·y
/// ẏ = δ·x·y - γ·y
/// ```
///
/// # Parameters
/// - `alpha`  — prey birth rate
/// - `beta`   — predation rate
/// - `gamma`  — predator death rate
/// - `delta`  — predator reproduction efficiency
/// - `t_span` — `(t_start, t_end)`
/// - `dt`     — integration step size
/// - `y0`     — initial state `[prey₀, predator₀]`
///
/// # Returns
/// `(t_vec, states)` where each element of `states` is `[prey, predator]`.
pub fn lotka_volterra(
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
    t_span: (f64, f64),
    dt: f64,
    y0: [f64; 2],
) -> Result<(Vec<f64>, Vec<[f64; 2]>)> {
    for (name, val) in [
        ("alpha", alpha),
        ("beta", beta),
        ("gamma", gamma),
        ("delta", delta),
    ] {
        if val < 0.0 {
            return Err(DatasetsError::InvalidFormat(format!(
                "lotka_volterra: {name} must be >= 0"
            )));
        }
    }
    if y0[0] < 0.0 || y0[1] < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "lotka_volterra: initial populations must be >= 0".into(),
        ));
    }
    rk4(
        |_t, y| {
            let dx = alpha * y[0] - beta * y[0] * y[1];
            let dy = delta * y[0] * y[1] - gamma * y[1];
            [dx, dy]
        },
        t_span,
        dt,
        y0,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenz 63
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Lorenz-63 attractor trajectory using RK4.
///
/// The equations are:
/// ```text
/// ẋ = σ·(y - x)
/// ẏ = x·(ρ - z) - y
/// ż = x·y - β·z
/// ```
///
/// Classic chaotic parameters: `sigma = 10`, `rho = 28`, `beta = 8/3`.
///
/// # Returns
/// `(t_vec, states)` where each element of `states` is `[x, y, z]`.
pub fn lorenz63(
    sigma: f64,
    rho: f64,
    beta: f64,
    t_span: (f64, f64),
    dt: f64,
    y0: [f64; 3],
) -> Result<(Vec<f64>, Vec<[f64; 3]>)> {
    rk4(
        |_t, y| {
            let dx = sigma * (y[1] - y[0]);
            let dy = y[0] * (rho - y[2]) - y[1];
            let dz = y[0] * y[1] - beta * y[2];
            [dx, dy, dz]
        },
        t_span,
        dt,
        y0,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Rössler attractor
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Rössler attractor trajectory.
///
/// The equations are:
/// ```text
/// ẋ = -y - z
/// ẏ = x + a·y
/// ż = b + z·(x - c)
/// ```
///
/// Classic parameters: `a = 0.2`, `b = 0.2`, `c = 5.7`.
///
/// # Returns
/// `(t_vec, states)` where each element of `states` is `[x, y, z]`.
pub fn roessler(
    a: f64,
    b: f64,
    c: f64,
    t_span: (f64, f64),
    dt: f64,
    y0: [f64; 3],
) -> Result<(Vec<f64>, Vec<[f64; 3]>)> {
    rk4(
        |_t, y| {
            let dx = -y[1] - y[2];
            let dy = y[0] + a * y[1];
            let dz = b + y[2] * (y[0] - c);
            [dx, dy, dz]
        },
        t_span,
        dt,
        y0,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Duffing oscillator
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a forced Duffing oscillator trajectory.
///
/// The equations (as a first-order system) are:
/// ```text
/// ẏ₁ = y₂
/// ẏ₂ = -δ·y₂ - α·y₁ - β·y₁³ + γ·cos(ω·t)
/// ```
///
/// # Parameters
/// - `alpha`  — linear stiffness
/// - `beta`   — cubic stiffness (use negative value for double-well potential)
/// - `delta`  — damping coefficient
/// - `gamma`  — forcing amplitude
/// - `omega`  — forcing angular frequency
///
/// # Returns
/// `(t_vec, states)` where each element of `states` is `[x, ẋ]`.
pub fn duffing_ode(
    alpha: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
    omega: f64,
    t_span: (f64, f64),
    dt: f64,
    y0: [f64; 2],
) -> Result<(Vec<f64>, Vec<[f64; 2]>)> {
    rk4(
        |t, y| {
            let dy1 = y[1];
            let dy2 = -delta * y[1] - alpha * y[0] - beta * y[0].powi(3)
                + gamma * (omega * t).cos();
            [dy1, dy2]
        },
        t_span,
        dt,
        y0,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple pendulum
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a simple pendulum trajectory (exact nonlinear equations).
///
/// The equations are:
/// ```text
/// θ̇ = ω
/// ω̇ = -(g/l)·sin(θ)
/// ```
///
/// # Parameters
/// - `l`      — pendulum length in metres (must be > 0)
/// - `g`      — gravitational acceleration (default 9.81 m/s²)
/// - `t_span` — `(t_start, t_end)`
/// - `dt`     — integration step size
/// - `y0`     — initial state `[θ₀ (rad), ω₀ (rad/s)]`
///
/// # Returns
/// `(t_vec, states)` where each element of `states` is `[θ, ω]`.
pub fn pendulum(
    l: f64,
    g: f64,
    t_span: (f64, f64),
    dt: f64,
    y0: [f64; 2],
) -> Result<(Vec<f64>, Vec<[f64; 2]>)> {
    if l <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "pendulum: l must be > 0".into(),
        ));
    }
    if g <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "pendulum: g must be > 0".into(),
        ));
    }
    let ratio = g / l;
    rk4(
        |_t, y| {
            let dtheta = y[1];
            let domega = -ratio * y[0].sin();
            [dtheta, domega]
        },
        t_span,
        dt,
        y0,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Double pendulum
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a double pendulum trajectory via Lagrangian mechanics.
///
/// State vector: `[θ₁, θ₂, ω₁, ω₂]` where `ω_i = dθ_i/dt`.
///
/// The full equations of motion derived from the Euler-Lagrange equations are:
/// ```text
/// ω̇₁ = [-g(2m₁+m₂)sin(θ₁) - m₂g·sin(θ₁-2θ₂) - 2sin(θ₁-θ₂)m₂(ω₂²l₂+ω₁²l₁cos(θ₁-θ₂))]
///       / [l₁(2m₁+m₂-m₂cos(2(θ₁-θ₂)))]
/// ω̇₂ = [2sin(θ₁-θ₂)(ω₁²l₁(m₁+m₂)+g(m₁+m₂)cos(θ₁)+ω₂²l₂m₂cos(θ₁-θ₂))]
///       / [l₂(2m₁+m₂-m₂cos(2(θ₁-θ₂)))]
/// ```
///
/// # Parameters
/// - `m1`, `m2` — masses (must be > 0)
/// - `l1`, `l2` — lengths (must be > 0)
/// - `g`        — gravitational acceleration
/// - `t_span`   — `(t_start, t_end)`
/// - `dt`       — step size
/// - `y0`       — initial state `[θ₁, θ₂, ω₁, ω₂]`
///
/// # Returns
/// `(t_vec, states)` where each element of `states` is `[θ₁, θ₂, ω₁, ω₂]`.
pub fn double_pendulum_ode(
    m1: f64,
    m2: f64,
    l1: f64,
    l2: f64,
    g: f64,
    t_span: (f64, f64),
    dt: f64,
    y0: [f64; 4],
) -> Result<(Vec<f64>, Vec<[f64; 4]>)> {
    for (name, val) in [("m1", m1), ("m2", m2), ("l1", l1), ("l2", l2)] {
        if val <= 0.0 {
            return Err(DatasetsError::InvalidFormat(format!(
                "double_pendulum_ode: {name} must be > 0"
            )));
        }
    }
    if g <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "double_pendulum_ode: g must be > 0".into(),
        ));
    }

    rk4(
        |_t, y| {
            let (t1, t2, w1, w2) = (y[0], y[1], y[2], y[3]);
            let dtheta = t1 - t2;
            let denom = 2.0 * m1 + m2 - m2 * (2.0 * dtheta).cos();

            let dw1_num = -g * (2.0 * m1 + m2) * t1.sin()
                - m2 * g * (t1 - 2.0 * t2).sin()
                - 2.0
                    * dtheta.sin()
                    * m2
                    * (w2 * w2 * l2 + w1 * w1 * l1 * dtheta.cos());
            let dw2_num = 2.0
                * dtheta.sin()
                * (w1 * w1 * l1 * (m1 + m2)
                    + g * (m1 + m2) * t1.cos()
                    + w2 * w2 * l2 * m2 * dtheta.cos());

            let dw1 = dw1_num / (l1 * denom);
            let dw2 = dw2_num / (l2 * denom);

            [w1, w2, dw1, dw2]
        },
        t_span,
        dt,
        y0,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_van_der_pol_ode_returns_correct_length() {
        let (t, s) = van_der_pol_ode(1.0, (0.0, 5.0), 0.1, [2.0, 0.0]).expect("valid params");
        assert!(!t.is_empty());
        assert_eq!(t.len(), s.len());
    }

    #[test]
    fn test_van_der_pol_ode_negative_mu_err() {
        assert!(van_der_pol_ode(-1.0, (0.0, 1.0), 0.1, [1.0, 0.0]).is_err());
    }

    #[test]
    fn test_lotka_volterra_conservation() {
        // Population must stay positive with valid params.
        let (_, s) = lotka_volterra(1.5, 1.0, 3.0, 1.0, (0.0, 10.0), 0.01, [10.0, 5.0]).expect("valid params");
        for state in &s {
            assert!(state[0].is_finite());
            assert!(state[1].is_finite());
        }
    }

    #[test]
    fn test_lorenz63_starts_at_y0() {
        let y0 = [1.0, 2.0, 3.0];
        let (_, s) = lorenz63(10.0, 28.0, 8.0 / 3.0, (0.0, 10.0), 0.01, y0).expect("valid params");
        assert_eq!(s[0], y0);
    }

    #[test]
    fn test_roessler_shape() {
        let (t, s) = roessler(0.2, 0.2, 5.7, (0.0, 5.0), 0.05, [1.0, 0.0, 0.0]).expect("valid params");
        assert_eq!(t.len(), s.len());
        assert!(t.len() > 10);
    }

    #[test]
    fn test_duffing_zero_forcing() {
        // With zero forcing a lightly damped oscillator should eventually decay.
        let (_, s) = duffing_ode(1.0, 0.0, 0.5, 0.0, 1.0, (0.0, 20.0), 0.01, [1.0, 0.0])
            .expect("valid params");
        let last = &s[s.len() - 1];
        assert!(last[0].abs() < 1.0, "amplitude should decay: {}", last[0]);
    }

    #[test]
    fn test_pendulum_small_angle_period() {
        // Small-angle period ≈ 2π√(l/g).
        let (l, g) = (1.0, 9.81);
        let expected_period = 2.0 * PI * (l / g).sqrt();
        let dt = 0.001;
        let t_end = expected_period * 3.0;
        let (t, s) = pendulum(l, g, (0.0, t_end), dt, [0.1, 0.0]).expect("valid params");

        // Count upward zero-crossings of θ.
        let mut crossings: Vec<f64> = vec![];
        for i in 1..t.len() {
            if s[i - 1][0] < 0.0 && s[i][0] >= 0.0 {
                crossings.push(t[i]);
            }
        }
        if crossings.len() >= 2 {
            let period = crossings[crossings.len() - 1] - crossings[crossings.len() - 2];
            let rel = (period - expected_period).abs() / expected_period;
            assert!(rel < 0.02, "period={period:.4}, expected≈{expected_period:.4}");
        }
    }

    #[test]
    fn test_double_pendulum_state_dim() {
        let y0 = [0.5, 0.5, 0.0, 0.0];
        let (t, s) =
            double_pendulum_ode(1.0, 1.0, 1.0, 1.0, 9.81, (0.0, 5.0), 0.01, y0).expect("valid params");
        assert_eq!(t.len(), s.len());
        for state in &s {
            for val in state {
                assert!(val.is_finite(), "non-finite value in double pendulum");
            }
        }
    }

    #[test]
    fn test_invalid_pendulum_l() {
        assert!(pendulum(0.0, 9.81, (0.0, 1.0), 0.01, [0.1, 0.0]).is_err());
    }

    #[test]
    fn test_invalid_dt() {
        assert!(lorenz63(10.0, 28.0, 2.667, (0.0, 1.0), -0.01, [1.0, 1.0, 1.0]).is_err());
    }
}
