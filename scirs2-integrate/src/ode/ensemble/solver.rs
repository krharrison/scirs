//! Dormand-Prince RK45 ensemble ODE solver with FSAL optimisation.
//!
//! Each ensemble member is solved independently with adaptive step control.
//! Members are distributed across threads using `std::thread::scope`.

use super::types::{EnsembleConfig, EnsembleResult};
use crate::error::{IntegrateError, IntegrateResult};

// ── Dormand-Prince RK45 Butcher tableau ──────────────────────────────────────
//
// Dormand, J.R.; Prince, P.J. (1980).  "A family of embedded Runge-Kutta
// formulae". J. Comput. Appl. Math. 6(1): 19-26.
//
// c2=1/5, c3=3/10, c4=4/5, c5=8/9, c6=1, c7=1
//
// The error estimate is  y5 − y4  using the 5th- and 4th-order solutions.

/// RK45 Dormand-Prince: a coefficients (6×5 lower-triangular)
const DP_A: [[f64; 6]; 6] = [
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [
        19_372.0 / 6_561.0,
        -25_360.0 / 2_187.0,
        64_448.0 / 6_561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
    ],
    [
        9_017.0 / 3_168.0,
        -355.0 / 33.0,
        46_732.0 / 5_247.0,
        49.0 / 176.0,
        -5_103.0 / 18_656.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1_113.0,
        125.0 / 192.0,
        -2_187.0 / 6_784.0,
        11.0 / 84.0,
    ],
];

/// RK45 Dormand-Prince: 5th-order weights (same as last row of A — FSAL)
const DP_B5: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1_113.0,
    125.0 / 192.0,
    -2_187.0 / 6_784.0,
    11.0 / 84.0,
    0.0,
];

/// RK45 Dormand-Prince: 4th-order embedded weights (for error estimate)
const DP_B4: [f64; 7] = [
    5_179.0 / 57_600.0,
    0.0,
    7_571.0 / 16_695.0,
    393.0 / 640.0,
    -92_097.0 / 339_200.0,
    187.0 / 2_100.0,
    1.0 / 40.0,
];

/// RK45 Dormand-Prince: node values c
const DP_C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

// ── Per-member integrator ─────────────────────────────────────────────────────

/// Internal state for integrating one ensemble member with RK45-FSAL.
struct Rk45State {
    t: f64,
    y: Vec<f64>,
    /// k1 (= f evaluated at current t, y) — reused from previous accepted step.
    k1: Vec<f64>,
    h: f64,
}

impl Rk45State {
    fn new(t0: f64, y0: Vec<f64>, k1: Vec<f64>, h_init: f64) -> Self {
        Self {
            t: t0,
            y: y0,
            k1,
            h: h_init,
        }
    }
}

/// Compute the RHS for a single stage.
fn stage<F, P>(f: &F, t: f64, y: &[f64], param: &P) -> Vec<f64>
where
    F: Fn(f64, &[f64], &P) -> Vec<f64>,
{
    f(t, y, param)
}

/// Add scaled vectors: `result[i] = a[i] + scale * b[i]`.
fn axpy(a: &[f64], scale: f64, b: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| ai + scale * bi)
        .collect()
}

/// Weighted sum: `result[i] = base[i] + h * Σ_j w[j] * ks[j][i]`.
fn rk_sum(base: &[f64], h: f64, weights: &[f64], ks: &[Vec<f64>]) -> Vec<f64> {
    let n = base.len();
    let mut result = base.to_vec();
    for (w, k) in weights.iter().zip(ks.iter()) {
        if w.abs() < f64::EPSILON {
            continue;
        }
        for i in 0..n {
            result[i] += h * w * k[i];
        }
    }
    result
}

/// Compute step-size error norm using mixed absolute/relative tolerance.
fn error_norm(y: &[f64], y_new: &[f64], e: &[f64], rtol: f64, atol: f64) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..n {
        let scale = atol + rtol * y[i].abs().max(y_new[i].abs());
        let ei = e[i] / scale;
        sum += ei * ei;
    }
    (sum / n as f64).sqrt()
}

/// Take one adaptive RK45-FSAL step.
///
/// Returns `(y_new, k1_new, h_next, accepted)`.
fn rk45_step<F, P>(
    f: &F,
    state: &Rk45State,
    t_end: f64,
    rtol: f64,
    atol: f64,
    param: &P,
) -> (Vec<f64>, Vec<f64>, f64, bool)
where
    F: Fn(f64, &[f64], &P) -> Vec<f64>,
{
    let t = state.t;
    let y = &state.y;
    let h = state.h.min(t_end - t);

    // Stages k1..k7
    let k1 = state.k1.clone();
    let y2 = axpy(y, h * DP_A[0][0], &k1);
    let k2 = stage(f, t + DP_C[1] * h, &y2, param);

    let y3 = {
        let mut v = y.to_vec();
        for i in 0..v.len() {
            v[i] += h * (DP_A[1][0] * k1[i] + DP_A[1][1] * k2[i]);
        }
        v
    };
    let k3 = stage(f, t + DP_C[2] * h, &y3, param);

    let y4 = {
        let mut v = y.to_vec();
        for i in 0..v.len() {
            v[i] += h * (DP_A[2][0] * k1[i] + DP_A[2][1] * k2[i] + DP_A[2][2] * k3[i]);
        }
        v
    };
    let k4 = stage(f, t + DP_C[3] * h, &y4, param);

    let y5 = {
        let mut v = y.to_vec();
        for i in 0..v.len() {
            v[i] += h
                * (DP_A[3][0] * k1[i]
                    + DP_A[3][1] * k2[i]
                    + DP_A[3][2] * k3[i]
                    + DP_A[3][3] * k4[i]);
        }
        v
    };
    let k5 = stage(f, t + DP_C[4] * h, &y5, param);

    let y6 = {
        let mut v = y.to_vec();
        for i in 0..v.len() {
            v[i] += h
                * (DP_A[4][0] * k1[i]
                    + DP_A[4][1] * k2[i]
                    + DP_A[4][2] * k3[i]
                    + DP_A[4][3] * k4[i]
                    + DP_A[4][4] * k5[i]);
        }
        v
    };
    let k6 = stage(f, t + DP_C[5] * h, &y6, param);

    // 5th-order solution (FSAL: k7 = f(t+h, y6) = next k1)
    let y_new = rk_sum(
        y,
        h,
        &DP_B5[..6],
        &[
            k1.clone(),
            k2.clone(),
            k3.clone(),
            k4.clone(),
            k5.clone(),
            k6.clone(),
        ],
    );
    let k7 = stage(f, t + h, &y_new, param);

    // 4th-order solution for error estimate
    let y4_ord = rk_sum(y, h, &DP_B4, &[k1, k2, k3, k4, k5, k6, k7.clone()]);

    // Error = 5th - 4th
    let e: Vec<f64> = y_new
        .iter()
        .zip(y4_ord.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let err = error_norm(y, &y_new, &e, rtol, atol);

    // Step-size control (PI controller, safety factor 0.9)
    let factor = if err == 0.0 {
        5.0
    } else {
        0.9 * err.powf(-0.2)
    };
    let factor = factor.clamp(0.2, 5.0);
    let h_next = h * factor;

    if err <= 1.0 {
        // Accepted
        (y_new, k7, h_next, true)
    } else {
        // Rejected — return unchanged y, propose smaller h
        (y.clone(), k7, h_next, false)
    }
}

/// Integrate a single ODE member from `t0` to `t_end`.
fn integrate_member<F, P>(
    f: &F,
    t0: f64,
    t_end: f64,
    y0: Vec<f64>,
    param: &P,
    rtol: f64,
    atol: f64,
    h_init: f64,
    max_steps: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, bool, usize)
where
    F: Fn(f64, &[f64], &P) -> Vec<f64>,
{
    let n_state = y0.len();

    // Choose initial step size
    let h0 = if h_init > 0.0 {
        h_init
    } else {
        // Estimate: h ~ 0.01 * (t_end - t0) but bounded
        ((t_end - t0) * 0.01).max(1e-8).min((t_end - t0) / 10.0)
    };

    let k1_0 = f(t0, &y0, param);
    let mut state = Rk45State::new(t0, y0.clone(), k1_0, h0);

    let mut traj = vec![y0];
    let mut times = vec![t0];
    let mut n_steps = 0_usize;

    while state.t < t_end - 1e-14 * (t_end - t0) && n_steps < max_steps {
        let (y_new, k_new, h_next, accepted) = rk45_step(f, &state, t_end, rtol, atol, param);

        if accepted {
            state.t = (state.t + state.h).min(t_end);
            state.y = y_new.clone();
            state.k1 = k_new;
            state.h = h_next.max(1e-14);
            n_steps += 1;
            traj.push(y_new);
            times.push(state.t);
        } else {
            // Step rejected; update step size only
            state.h = h_next.max(1e-14);
        }

        // Avoid step-size going below machine epsilon
        if state.h < 1e-14 * state.t.abs().max(1.0) {
            break;
        }
    }

    let converged = if (state.t - t_end).abs() < 1e-12 * (t_end - t0 + 1.0) {
        true
    } else if n_steps == max_steps {
        // Reached max; not fully converged
        false
    } else {
        state.t >= t_end - 1e-10 * (t_end - t0)
    };

    // Ensure state vector isn't empty
    if traj.is_empty() {
        traj.push(vec![0.0; n_state]);
        times.push(t0);
    }

    (traj, times, converged, n_steps)
}

// ── Public solver ─────────────────────────────────────────────────────────────

/// Batched parallel ODE ensemble integrator.
///
/// Solves `n_ensemble` ODE IVPs in parallel.  Each member may have different
/// initial conditions and/or parameters.
pub struct OdeEnsembleSolver {
    /// Configuration for the ensemble.
    pub config: EnsembleConfig,
}

impl OdeEnsembleSolver {
    /// Create a new solver with the given configuration.
    pub fn new(config: EnsembleConfig) -> Self {
        Self { config }
    }

    /// Integrate the ensemble.
    ///
    /// # Type parameters
    ///
    /// * `F` — RHS function `f(t, y, &param) -> Vec<f64>`.  Must be `Fn + Sync`.
    /// * `P` — Parameter type.  Must be `Sync`.
    ///
    /// # Arguments
    ///
    /// * `f`       - ODE right-hand side.
    /// * `params`  - Slice of parameters, one per member.
    /// * `y0s`     - Slice of initial conditions, one per member.
    /// * `config`  - Ensemble configuration (can differ from `self.config`).
    ///
    /// # Errors
    ///
    /// Returns `IntegrateError::InvalidInput` if `params.len() != y0s.len()`
    /// or if `t_span` is invalid.
    pub fn solve<F, P>(
        &self,
        f: F,
        params: &[P],
        y0s: &[Vec<f64>],
        config: &EnsembleConfig,
    ) -> IntegrateResult<EnsembleResult>
    where
        F: Fn(f64, &[f64], &P) -> Vec<f64> + Sync,
        P: Sync,
    {
        if params.len() != y0s.len() {
            return Err(IntegrateError::InvalidInput(format!(
                "params.len() ({}) != y0s.len() ({})",
                params.len(),
                y0s.len()
            )));
        }

        let (t0, t_end) = config.t_span;
        if t0 >= t_end {
            return Err(IntegrateError::InvalidInput(
                "t_span must satisfy t0 < t_end".to_string(),
            ));
        }

        let n = params.len();
        if n == 0 {
            return Ok(EnsembleResult {
                trajectories: vec![],
                times: vec![],
                converged: vec![],
                n_steps: vec![],
            });
        }

        let rtol = config.rtol;
        let atol = config.atol;
        let h_init = config.h_init;
        let max_steps = config.max_steps;
        let n_threads = config.n_threads.max(1).min(n);

        // Pre-allocate result storage
        let mut trajectories: Vec<Vec<Vec<f64>>> = vec![Vec::new(); n];
        let mut times_out: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut converged: Vec<bool> = vec![false; n];
        let mut n_steps_out: Vec<usize> = vec![0; n];

        // We use indices to distribute work across threads.
        // Build index chunks.
        let chunk_size = n.div_ceil(n_threads);

        // Shared result slots — declared OUTSIDE scope so they outlive the threads.
        let results: Vec<std::sync::Mutex<Option<(Vec<Vec<f64>>, Vec<f64>, bool, usize)>>> =
            (0..n).map(|_| std::sync::Mutex::new(None)).collect();

        // Use thread::scope so we can borrow f, params, y0s safely.
        std::thread::scope(|scope| {
            let results_ref = &results;
            let f_ref = &f;

            // Spawn one thread per chunk.
            for tid in 0..n_threads {
                let start = tid * chunk_size;
                if start >= n {
                    break;
                }
                let end = (start + chunk_size).min(n);
                let params_slice = &params[start..end];
                let y0s_slice = &y0s[start..end];

                scope.spawn(move || {
                    for (local_idx, (param, y0)) in
                        params_slice.iter().zip(y0s_slice.iter()).enumerate()
                    {
                        let global_idx = start + local_idx;
                        let (traj, ts, conv, ns) = integrate_member(
                            f_ref,
                            t0,
                            t_end,
                            y0.clone(),
                            param,
                            rtol,
                            atol,
                            h_init,
                            max_steps,
                        );
                        // Write result
                        if let Ok(mut slot) = results_ref[global_idx].lock() {
                            *slot = Some((traj, ts, conv, ns));
                        }
                    }
                });
            }
            // scope drops here, joining all spawned threads.
        });

        // Collect results
        for (i, slot) in results.into_iter().enumerate() {
            if let Ok(Some((traj, ts, conv, ns))) = slot.into_inner() {
                trajectories[i] = traj;
                times_out[i] = ts;
                converged[i] = conv;
                n_steps_out[i] = ns;
            }
        }

        Ok(EnsembleResult {
            trajectories,
            times: times_out,
            converged,
            n_steps: n_steps_out,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// dx/dt = -x, x(0)=1 → x(t) = e^{-t}.  Check 10 identical members.
    #[test]
    fn test_ensemble_exponential_decay() {
        let cfg = EnsembleConfig {
            n_ensemble: 10,
            n_threads: 2,
            rtol: 1e-8,
            atol: 1e-10,
            t_span: (0.0, 1.0),
            max_steps: 10_000,
            h_init: 0.0,
        };

        let solver = OdeEnsembleSolver::new(cfg.clone());
        let params: Vec<f64> = vec![1.0; 10];
        let y0s: Vec<Vec<f64>> = vec![vec![1.0]; 10];

        let result = solver
            .solve(|_t, y, &p| vec![-p * y[0]], &params, &y0s, &cfg)
            .expect("solve failed");

        assert_eq!(result.trajectories.len(), 10);
        for (i, (traj, ts)) in result
            .trajectories
            .iter()
            .zip(result.times.iter())
            .enumerate()
        {
            let t_final = *ts.last().expect("no times");
            let y_final = traj.last().expect("no trajectory")[0];
            let expected = (-t_final).exp();
            assert!(
                approx_eq(y_final, expected, 1e-5),
                "member {i}: y(t={t_final:.4}) = {y_final:.8}, expected {expected:.8}"
            );
        }
    }

    /// All 10 members should converge for the simple decay ODE.
    #[test]
    fn test_ensemble_all_converged() {
        let cfg = EnsembleConfig {
            n_ensemble: 10,
            n_threads: 4,
            rtol: 1e-8,
            atol: 1e-10,
            t_span: (0.0, 2.0),
            max_steps: 50_000,
            h_init: 0.0,
        };
        let solver = OdeEnsembleSolver::new(cfg.clone());
        let params: Vec<f64> = vec![1.0; 10];
        let y0s: Vec<Vec<f64>> = vec![vec![1.0]; 10];

        let result = solver
            .solve(|_t, y, &p| vec![-p * y[0]], &params, &y0s, &cfg)
            .expect("solve failed");

        for (i, &conv) in result.converged.iter().enumerate() {
            assert!(conv, "member {i} did not converge");
        }
    }

    /// Different initial conditions lead to different final values.
    #[test]
    fn test_ensemble_different_ics() {
        let cfg = EnsembleConfig {
            n_ensemble: 5,
            n_threads: 2,
            rtol: 1e-8,
            atol: 1e-10,
            t_span: (0.0, 1.0),
            max_steps: 10_000,
            h_init: 0.0,
        };
        let solver = OdeEnsembleSolver::new(cfg.clone());
        let params: Vec<f64> = vec![1.0; 5];
        // y0 = 1.0, 2.0, 3.0, 4.0, 5.0
        let y0s: Vec<Vec<f64>> = (1..=5).map(|i| vec![i as f64]).collect();

        let result = solver
            .solve(|_t, y, &p| vec![-p * y[0]], &params, &y0s, &cfg)
            .expect("solve failed");

        // Final values should differ
        let finals: Vec<f64> = result
            .trajectories
            .iter()
            .map(|traj| traj.last().expect("no traj")[0])
            .collect();

        for i in 1..finals.len() {
            assert!(
                (finals[i] - finals[0]).abs() > 0.1,
                "members 0 and {i} should differ: {} vs {}",
                finals[0],
                finals[i]
            );
        }
    }

    /// EnsembleConfig::default() is well-formed.
    #[test]
    fn test_ensemble_config_default() {
        let cfg = EnsembleConfig::default();
        assert!(cfg.n_ensemble > 0);
        assert!(cfg.n_threads > 0);
        assert!(cfg.rtol > 0.0);
        assert!(cfg.atol > 0.0);
        let (t0, t1) = cfg.t_span;
        assert!(t0 < t1);
    }

    /// n_threads=1 vs n_threads=2 should give identical results.
    #[test]
    fn test_ensemble_parallel_same_as_serial() {
        let mk_cfg = |n_threads: usize| EnsembleConfig {
            n_ensemble: 4,
            n_threads,
            rtol: 1e-8,
            atol: 1e-10,
            t_span: (0.0, 1.0),
            max_steps: 10_000,
            h_init: 0.0,
        };

        let params: Vec<f64> = vec![0.5, 1.0, 1.5, 2.0];
        let y0s: Vec<Vec<f64>> = vec![vec![1.0]; 4];

        let f = |_t: f64, y: &[f64], &p: &f64| vec![-p * y[0]];

        let cfg1 = mk_cfg(1);
        let solver1 = OdeEnsembleSolver::new(cfg1.clone());
        let res1 = solver1
            .solve(f, &params, &y0s, &cfg1)
            .expect("solve 1 failed");

        let cfg2 = mk_cfg(2);
        let solver2 = OdeEnsembleSolver::new(cfg2.clone());
        let res2 = solver2
            .solve(f, &params, &y0s, &cfg2)
            .expect("solve 2 failed");

        for i in 0..4 {
            let y1 = res1.trajectories[i].last().expect("no traj1")[0];
            let y2 = res2.trajectories[i].last().expect("no traj2")[0];
            assert!(
                approx_eq(y1, y2, 1e-10),
                "member {i}: thread-1={y1}, thread-2={y2}"
            );
        }
    }

    /// Mean of identical ODEs should equal the single ODE solution.
    #[test]
    fn test_ensemble_mean_trajectory() {
        let cfg = EnsembleConfig {
            n_ensemble: 5,
            n_threads: 2,
            rtol: 1e-8,
            atol: 1e-10,
            t_span: (0.0, 1.0),
            max_steps: 10_000,
            h_init: 1e-3,
        };
        let solver = OdeEnsembleSolver::new(cfg.clone());
        let params: Vec<f64> = vec![1.0; 5];
        let y0s: Vec<Vec<f64>> = vec![vec![1.0]; 5];

        let result = solver
            .solve(|_t, y, &p| vec![-p * y[0]], &params, &y0s, &cfg)
            .expect("solve failed");

        let mean = result.mean_trajectory().expect("mean failed");
        // Mean should equal single trajectory (all identical)
        let single = &result.trajectories[0];
        let min_len = mean.len().min(single.len());
        for k in 0..min_len {
            assert!(
                approx_eq(mean[k][0], single[k][0], 1e-10),
                "mean[{k}]={}, single[{k}]={}",
                mean[k][0],
                single[k][0]
            );
        }
    }
}
