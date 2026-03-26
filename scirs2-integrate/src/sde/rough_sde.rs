//! Rough SDE solver driven by fractional Brownian Motion.
//!
//! A *rough SDE* is a stochastic differential equation of the form
//!
//! ```text
//! dX = f(t, X) dt + g(t, X) dB^H
//! ```
//!
//! where B^H is a fractional Brownian Motion with Hurst parameter H ∈ (0, 1).
//!
//! When H < 0.5 the driving noise is *rougher* than standard Brownian motion
//! (Hölder regularity < 0.5).  This regime is encountered in rough volatility
//! models (Gatheral-Jaisson-Rosenbaum 2018) where H ≈ 0.1.
//!
//! ## Numerical schemes
//!
//! | Scheme | Description |
//! |--------|-------------|
//! | `EulerMaruyama` | X_{n+1} = X_n + f dt + g ΔB^H |
//! | `Milstein` | Euler + Wong-Zakai correction ½ g g' (2H−1) dt^{2H−1} dt |
//!
//! ## Bergomi rough volatility model
//!
//! The convenience function [`bergomi_model`] implements a simplified one-factor
//! Bergomi model:
//!
//! ```text
//! log σ_t = ν B^H_t − ½ ν² t^{2H}
//! σ_t     = σ_0 exp(ν B^H_t − ½ ν² t^{2H})
//! ```
//!
//! ## References
//!
//! * El Euch, O., Rosenbaum, M. (2019). *The characteristic function of rough
//!   Heston models*, Mathematical Finance.
//! * Gatheral, J., Jaisson, T., Rosenbaum, M. (2018). *Volatility is rough*,
//!   Quantitative Finance.

use crate::error::{IntegrateError, IntegrateResult};
use crate::sde::fractional_brownian::{FbmConfig, FbmMethod, FractionalBrownianMotion};
use scirs2_core::ndarray::Array1;

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Numerical scheme for the rough SDE solver.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoughScheme {
    /// First-order Euler-Maruyama discretisation.
    ///
    /// ```text
    /// X_{n+1} = X_n + f(t_n, X_n) dt + g(t_n, X_n) ΔB^H_n
    /// ```
    EulerMaruyama,
    /// Milstein-type correction including the Wong-Zakai term for H ≠ 0.5.
    ///
    /// For H ≠ 0.5 the correction is
    /// ```text
    /// X_{n+1} = Euler + ½ g(t,X) g'(t,X) (2H−1) dt^{2H}
    /// ```
    /// where g' ≈ (g(t, x+ε) − g(t, x)) / ε is a finite-difference derivative.
    Milstein,
}

/// Configuration for the rough SDE solver.
#[derive(Debug, Clone)]
pub struct RoughSdeConfig {
    /// Hurst exponent H ∈ (0, 1).
    pub hurst: f64,
    /// Number of time steps.
    pub n_steps: usize,
    /// Terminal time T.
    pub t_end: f64,
    /// Numerical scheme.
    pub scheme: RoughScheme,
    /// Seed for the internal fBm PRNG.
    pub seed: u64,
}

impl Default for RoughSdeConfig {
    fn default() -> Self {
        Self {
            hurst: 0.7,
            n_steps: 500,
            t_end: 1.0,
            scheme: RoughScheme::EulerMaruyama,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Output of the rough SDE solver.
#[derive(Debug, Clone)]
pub struct RoughSdeResult {
    /// Discrete time grid t_0 = 0, t_1, …, t_{n_steps} = t_end.
    pub times: Array1<f64>,
    /// Numerical solution X_{t_0}, X_{t_1}, …, X_{t_{n_steps}}.
    pub path: Array1<f64>,
    /// fBm increments ΔB^H_0, …, ΔB^H_{n_steps − 1}.
    pub fbm_increments: Array1<f64>,
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Solve a rough SDE: dX = f(t, X) dt + g(t, X) dB^H.
///
/// # Arguments
///
/// * `f` — drift coefficient f(t, x) → ℝ
/// * `g` — diffusion coefficient g(t, x) → ℝ
/// * `x0` — initial value X(0)
/// * `config` — solver configuration (Hurst exponent, steps, scheme, …)
///
/// # Returns
///
/// A [`RoughSdeResult`] containing the time grid, solution path, and fBm
/// increments used.
///
/// # Errors
///
/// Returns [`IntegrateError`] if the configuration is invalid or the fBm
/// generator fails.
///
/// # Example
///
/// ```rust
/// use scirs2_integrate::sde::rough_sde::{rough_sde_solve, RoughSdeConfig, RoughScheme};
///
/// // Geometric rough SDE: dX = 0 dt + X dB^H  (pure fBm integral)
/// let cfg = RoughSdeConfig {
///     hurst: 0.7, n_steps: 128, t_end: 1.0,
///     scheme: RoughScheme::EulerMaruyama, seed: 0,
/// };
/// let res = rough_sde_solve(
///     |_t, _x| 0.0,
///     |_t, x| x,
///     1.0,
///     &cfg,
/// ).unwrap();
/// assert_eq!(res.path.len(), 129); // n_steps + 1
/// ```
pub fn rough_sde_solve<F, G>(
    f: F,
    g: G,
    x0: f64,
    config: &RoughSdeConfig,
) -> IntegrateResult<RoughSdeResult>
where
    F: Fn(f64, f64) -> f64,
    G: Fn(f64, f64) -> f64,
{
    validate_config(config)?;

    let n = config.n_steps;
    let dt = config.t_end / n as f64;

    // Generate fBm increments ΔB^H
    let fbm_cfg = FbmConfig {
        hurst: config.hurst,
        n_steps: n,
        dt,
        seed: config.seed,
        method: FbmMethod::DaviesHarte,
    };
    let fbm = FractionalBrownianMotion::new(fbm_cfg);
    let db = fbm.increments().map_err(|e| {
        IntegrateError::ComputationError(format!("fBm increment generation failed: {e}"))
    })?;

    // Build time grid
    let times_vec: Vec<f64> = (0..=n).map(|i| i as f64 * dt).collect();
    let times = Array1::from_vec(times_vec);

    // Time-step integration
    let mut path = Array1::zeros(n + 1);
    path[0] = x0;

    let h = config.hurst;
    // Wong-Zakai coefficient used by Milstein scheme:
    // c_wz = ½ (2H − 1) dt^{2H}  (correction per step, independent of step index for const. dt)
    let wz_coeff = 0.5 * (2.0 * h - 1.0) * dt.powf(2.0 * h);
    // Finite-difference epsilon for g derivative
    let eps = 1e-6_f64;

    for i in 0..n {
        let t_i = times[i];
        let x_i = path[i];
        let db_i = db[i];

        let f_val = f(t_i, x_i);
        let g_val = g(t_i, x_i);

        let euler_step = f_val * dt + g_val * db_i;

        #[allow(unreachable_patterns)]
        let correction = match config.scheme {
            RoughScheme::EulerMaruyama => 0.0,
            RoughScheme::Milstein => {
                // g'(t, x) ≈ (g(t, x+ε) − g(t, x)) / ε
                let g_plus = g(t_i, x_i + eps);
                let g_prime = (g_plus - g_val) / eps;
                g_val * g_prime * wz_coeff
            }
            // Future variants of RoughScheme (non_exhaustive guard)
            _ => 0.0,
        };

        path[i + 1] = x_i + euler_step + correction;
    }

    Ok(RoughSdeResult {
        times,
        path,
        fbm_increments: db,
    })
}

// ---------------------------------------------------------------------------
// Bergomi rough volatility model
// ---------------------------------------------------------------------------

/// Simplified one-factor Bergomi rough volatility model.
///
/// Models the instantaneous variance (squared volatility) as
///
/// ```text
/// v_t = v_0 exp(ν B^H_t − ½ ν² t^{2H})
/// ```
///
/// and returns the volatility σ_t = √v_t together with the time grid.
///
/// # Arguments
///
/// * `nu` — volatility-of-volatility parameter ν
/// * `v0` — initial variance (not volatility) v_0 > 0
/// * `hurst` — Hurst exponent H (typical rough volatility: H ≈ 0.1)
/// * `n_steps` — number of time discretisation steps
/// * `t_end` — terminal time T
/// * `seed` — PRNG seed
///
/// # Returns
///
/// A pair `(times, vol_path)` where `vol_path[i] = σ_{t_i} = √v_{t_i}`.
///
/// # Errors
///
/// Returns [`IntegrateError`] if parameters are invalid.
///
/// # Example
///
/// ```rust
/// use scirs2_integrate::sde::rough_sde::bergomi_model;
///
/// let (times, vol) = bergomi_model(0.3, 0.04, 0.1, 256, 1.0, 0).unwrap();
/// assert_eq!(vol.len(), 257);
/// assert!(vol.iter().all(|&v| v > 0.0), "volatility must be strictly positive");
/// ```
pub fn bergomi_model(
    nu: f64,
    v0: f64,
    hurst: f64,
    n_steps: usize,
    t_end: f64,
    seed: u64,
) -> IntegrateResult<(Array1<f64>, Array1<f64>)> {
    if v0 <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Initial variance v0 must be positive, got {}",
            v0
        )));
    }
    if hurst <= 0.0 || hurst >= 1.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Hurst exponent must be in (0, 1), got {}",
            hurst
        )));
    }
    if n_steps == 0 {
        return Err(IntegrateError::InvalidInput(
            "n_steps must be at least 1".to_string(),
        ));
    }
    if t_end <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "t_end must be positive, got {}",
            t_end
        )));
    }

    let dt = t_end / n_steps as f64;

    // Generate fBm path B^H_0, B^H_dt, …, B^H_{T} (length n_steps+1)
    let fbm_cfg = FbmConfig {
        hurst,
        n_steps,
        dt,
        seed,
        method: FbmMethod::DaviesHarte,
    };
    let fbm_path = FractionalBrownianMotion::new(fbm_cfg).sample_path().map_err(|e| {
        IntegrateError::ComputationError(format!("Bergomi fBm path failed: {e}"))
    })?;

    let h2 = 2.0 * hurst;

    // σ_t = √v_0 · exp( ν B^H_t − ½ ν² t^{2H} )
    let vol_path: Vec<f64> = (0..=n_steps)
        .map(|i| {
            let t_i = i as f64 * dt;
            let bh = fbm_path[i];
            let log_vol = 0.5 * v0.ln() + nu * bh - 0.5 * nu * nu * t_i.powf(h2);
            log_vol.exp()
        })
        .collect();

    let times: Vec<f64> = (0..=n_steps).map(|i| i as f64 * dt).collect();

    Ok((Array1::from_vec(times), Array1::from_vec(vol_path)))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_config(cfg: &RoughSdeConfig) -> IntegrateResult<()> {
    if cfg.hurst <= 0.0 || cfg.hurst >= 1.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Hurst exponent must be in (0, 1), got {}",
            cfg.hurst
        )));
    }
    if cfg.n_steps == 0 {
        return Err(IntegrateError::InvalidInput(
            "n_steps must be at least 1".to_string(),
        ));
    }
    if cfg.t_end <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "t_end must be positive, got {}",
            cfg.t_end
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> RoughSdeConfig {
        RoughSdeConfig::default()
    }

    fn em_cfg(h: f64, n: usize) -> RoughSdeConfig {
        RoughSdeConfig {
            hurst: h,
            n_steps: n,
            t_end: 1.0,
            scheme: RoughScheme::EulerMaruyama,
            seed: 42,
        }
    }

    #[test]
    fn test_default_config_runs() {
        let res = rough_sde_solve(
            |_t, _x| 0.0,
            |_t, _x| 1.0,
            0.0,
            &default_cfg(),
        )
        .expect("default config should succeed");
        assert_eq!(res.path.len(), default_cfg().n_steps + 1);
    }

    #[test]
    fn test_path_length() {
        let cfg = em_cfg(0.7, 200);
        let res = rough_sde_solve(|_t, _x| 0.5, |_t, _x| 0.1, 1.0, &cfg)
            .expect("solve should succeed");
        assert_eq!(res.path.len(), 201); // n_steps + 1
    }

    #[test]
    fn test_times_length_and_end_value() {
        let cfg = em_cfg(0.6, 100);
        let res = rough_sde_solve(|_t, _x| 0.0, |_t, _x| 0.0, 0.0, &cfg)
            .expect("solve should succeed");
        assert_eq!(res.times.len(), 101);
        let t_last = res.times[100];
        assert!(
            (t_last - 1.0).abs() < 1e-10,
            "last time = {}, expected 1.0",
            t_last
        );
    }

    #[test]
    fn test_zero_diffusion_matches_ode() {
        // g = 0 → X_n = x0 + sum(f * dt) = x0 + f * t_end  (for constant f)
        let f_val = 2.0_f64;
        let x0 = 1.0_f64;
        let n = 100;
        let t_end = 1.0;
        let cfg = em_cfg(0.7, n);
        let res = rough_sde_solve(
            move |_t, _x| f_val,
            |_t, _x| 0.0,
            x0,
            &cfg,
        )
        .expect("solve should succeed");

        let expected_end = x0 + f_val * t_end;
        let actual_end = res.path[n];
        let rel = (actual_end - expected_end).abs() / expected_end;
        assert!(
            rel < 1e-10,
            "zero-diffusion ODE: got {}, expected {}, rel err {}",
            actual_end, expected_end, rel
        );
    }

    #[test]
    fn test_zero_drift_pure_fbm_integral() {
        // f = 0, g = 1 → X is the fBm integral (path should be non-trivial, variance > 0)
        let cfg = em_cfg(0.7, 256);
        let res = rough_sde_solve(|_t, _x| 0.0, |_t, _x| 1.0, 0.0, &cfg)
            .expect("solve should succeed");
        // path[0] = 0
        assert!((res.path[0]).abs() < 1e-15);
        // path should have non-zero values (with overwhelming probability)
        let sum_sq: f64 = res.path.iter().map(|&v| v * v).sum();
        assert!(sum_sq > 1e-20, "zero-drift path should be non-trivial");
    }

    #[test]
    fn test_h_half_euler_matches_fbm_path() {
        // For H = 0.5, the rough EM with f=0, g=1 should equal the fBm path.
        let n = 128;
        let dt = 1.0 / n as f64;
        let seed = 0;
        let cfg = RoughSdeConfig {
            hurst: 0.5, n_steps: n, t_end: 1.0,
            scheme: RoughScheme::EulerMaruyama, seed,
        };
        let res = rough_sde_solve(|_t, _x| 0.0, |_t, _x| 1.0, 0.0, &cfg)
            .expect("solve");

        // Independently generate fBm path with same parameters
        let fbm_cfg = FbmConfig {
            hurst: 0.5, n_steps: n, dt, seed,
            method: FbmMethod::DaviesHarte,
        };
        let fbm_path = FractionalBrownianMotion::new(fbm_cfg)
            .sample_path()
            .expect("fbm path");

        // Both should equal the cumulative sum of the same increments.
        // The fbm increments are regenerated with the same seed, so they match.
        for i in 0..=n {
            assert!(
                (res.path[i] - fbm_path[i]).abs() < 1e-12,
                "mismatch at i={}: rough={}, fbm={}",
                i, res.path[i], fbm_path[i]
            );
        }
    }

    #[test]
    fn test_milstein_differs_from_euler_for_h_ne_half() {
        // For H ≠ 0.5 and non-zero g, Milstein correction is non-zero.
        let n = 128;
        let cfg_em = RoughSdeConfig {
            hurst: 0.3, n_steps: n, t_end: 1.0,
            scheme: RoughScheme::EulerMaruyama, seed: 99,
        };
        let cfg_mil = RoughSdeConfig {
            scheme: RoughScheme::Milstein, ..cfg_em.clone()
        };
        let res_em = rough_sde_solve(
            |_t, x| 0.1 * x, |_t, x| 0.2 * x, 1.0, &cfg_em,
        )
        .expect("EM solve");
        let res_mil = rough_sde_solve(
            |_t, x| 0.1 * x, |_t, x| 0.2 * x, 1.0, &cfg_mil,
        )
        .expect("Milstein solve");

        // Results must differ (correction is non-zero for H=0.3)
        let max_diff: f64 = res_em
            .path
            .iter()
            .zip(res_mil.path.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-12,
            "Milstein and Euler should differ for H=0.3, max_diff={}",
            max_diff
        );
    }

    #[test]
    fn test_bergomi_vol_always_positive() {
        let (times, vol) = bergomi_model(0.3, 0.04, 0.1, 256, 1.0, 0)
            .expect("bergomi_model should succeed");
        assert_eq!(vol.len(), 257);
        assert_eq!(times.len(), 257);
        for (i, &v) in vol.iter().enumerate() {
            assert!(
                v > 0.0,
                "volatility must be strictly positive at step {}, got {}",
                i, v
            );
        }
    }

    #[test]
    fn test_result_times_length() {
        let cfg = em_cfg(0.7, 50);
        let res = rough_sde_solve(|_t, _x| 0.0, |_t, _x| 1.0, 0.5, &cfg)
            .expect("solve");
        assert_eq!(res.times.len(), 51);
        assert_eq!(res.fbm_increments.len(), 50);
    }

    #[test]
    fn test_trivial_single_step() {
        let cfg = RoughSdeConfig {
            hurst: 0.7, n_steps: 1, t_end: 0.1,
            scheme: RoughScheme::EulerMaruyama, seed: 0,
        };
        let res = rough_sde_solve(|_t, _x| 1.0, |_t, _x| 0.0, 2.0, &cfg)
            .expect("single-step solve");
        assert_eq!(res.path.len(), 2);
        // path[1] = 2.0 + 1.0 * 0.1 = 2.1
        let expected = 2.0 + 1.0 * 0.1;
        assert!(
            (res.path[1] - expected).abs() < 1e-10,
            "single-step: got {}, expected {}",
            res.path[1], expected
        );
    }
}
