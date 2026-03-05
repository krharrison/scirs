//! Symplectic integrators for Hamiltonian systems within the ODE framework
//!
//! This module provides advanced symplectic integration methods that preserve
//! the geometric structure of Hamiltonian systems. These are particularly useful
//! for long-time integration of conservative systems where energy conservation
//! is critical.
//!
//! # Methods provided
//!
//! - **Stormer-Verlet (Leapfrog)**: 2nd order, most widely used
//! - **Velocity Verlet**: Variant optimized for molecular dynamics
//! - **Yoshida 4th order**: Higher accuracy via triple-jump composition
//! - **Yoshida 6th order**: Even higher accuracy, 7-stage composition
//! - **Yoshida 8th order**: Very high accuracy, 15-stage composition
//!
//! # Energy monitoring
//!
//! All integrators track energy drift via the `EnergyMonitor` which records
//! per-step and cumulative energy errors, enabling early detection of numerical
//! instabilities.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::Array1;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Hamiltonian system trait (ODE-methods flavour, independent of symplectic/)
// ---------------------------------------------------------------------------

/// Trait representing a Hamiltonian system for symplectic integration.
///
/// The system is described by Hamilton's equations:
///   dq/dt =  dH/dp
///   dp/dt = -dH/dq
pub trait HamiltonianSystem<F: IntegrateFloat> {
    /// Number of degrees of freedom (dimension of q or p).
    fn ndof(&self) -> usize;

    /// Compute dq/dt = dH/dp evaluated at (t, q, p).
    fn dq_dt(&self, t: F, q: &Array1<F>, p: &Array1<F>) -> IntegrateResult<Array1<F>>;

    /// Compute dp/dt = -dH/dq evaluated at (t, q, p).
    fn dp_dt(&self, t: F, q: &Array1<F>, p: &Array1<F>) -> IntegrateResult<Array1<F>>;

    /// Optionally compute the Hamiltonian H(t, q, p) for energy monitoring.
    /// Returns `None` if no closed-form Hamiltonian is available.
    fn hamiltonian(&self, _t: F, _q: &Array1<F>, _p: &Array1<F>) -> Option<F> {
        None
    }
}

/// A separable Hamiltonian H(q, p) = T(p) + V(q).
///
/// For separable systems the equations of motion are:
///   dq/dt = dT/dp
///   dp/dt = -dV/dq
pub struct SeparableSystem<F: IntegrateFloat> {
    ndof: usize,
    /// dT/dp (kinetic gradient)
    kinetic_grad: Box<dyn Fn(F, &Array1<F>) -> Array1<F> + Send + Sync>,
    /// dV/dq (potential gradient)
    potential_grad: Box<dyn Fn(F, &Array1<F>) -> Array1<F> + Send + Sync>,
    /// Optional: T(p) for energy monitoring
    kinetic_energy: Option<Box<dyn Fn(F, &Array1<F>) -> F + Send + Sync>>,
    /// Optional: V(q) for energy monitoring
    potential_energy: Option<Box<dyn Fn(F, &Array1<F>) -> F + Send + Sync>>,
}

impl<F: IntegrateFloat> SeparableSystem<F> {
    /// Create a separable system from gradient functions.
    ///
    /// # Arguments
    /// * `ndof` - number of degrees of freedom
    /// * `kinetic_grad` - computes dT/dp
    /// * `potential_grad` - computes dV/dq
    pub fn new<KG, VG>(ndof: usize, kinetic_grad: KG, potential_grad: VG) -> Self
    where
        KG: Fn(F, &Array1<F>) -> Array1<F> + Send + Sync + 'static,
        VG: Fn(F, &Array1<F>) -> Array1<F> + Send + Sync + 'static,
    {
        SeparableSystem {
            ndof,
            kinetic_grad: Box::new(kinetic_grad),
            potential_grad: Box::new(potential_grad),
            kinetic_energy: None,
            potential_energy: None,
        }
    }

    /// Attach energy functions for monitoring.
    pub fn with_energy<KE, VE>(mut self, kinetic_energy: KE, potential_energy: VE) -> Self
    where
        KE: Fn(F, &Array1<F>) -> F + Send + Sync + 'static,
        VE: Fn(F, &Array1<F>) -> F + Send + Sync + 'static,
    {
        self.kinetic_energy = Some(Box::new(kinetic_energy));
        self.potential_energy = Some(Box::new(potential_energy));
        self
    }
}

impl<F: IntegrateFloat> HamiltonianSystem<F> for SeparableSystem<F> {
    fn ndof(&self) -> usize {
        self.ndof
    }

    fn dq_dt(&self, t: F, _q: &Array1<F>, p: &Array1<F>) -> IntegrateResult<Array1<F>> {
        Ok((self.kinetic_grad)(t, p))
    }

    fn dp_dt(&self, t: F, q: &Array1<F>, _p: &Array1<F>) -> IntegrateResult<Array1<F>> {
        // dp/dt = -dV/dq
        let grad_v = (self.potential_grad)(t, q);
        Ok(grad_v.mapv(|x| -x))
    }

    fn hamiltonian(&self, t: F, q: &Array1<F>, p: &Array1<F>) -> Option<F> {
        match (&self.kinetic_energy, &self.potential_energy) {
            (Some(ke), Some(ve)) => Some(ke(t, p) + ve(t, q)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Energy monitor
// ---------------------------------------------------------------------------

/// Tracks energy drift during symplectic integration.
#[derive(Debug, Clone)]
pub struct EnergyMonitor<F: IntegrateFloat> {
    /// Initial energy H_0
    pub initial_energy: Option<F>,
    /// Energy at each recorded step
    pub energy_history: Vec<F>,
    /// Absolute energy error |H(t) - H_0| at each recorded step
    pub abs_errors: Vec<F>,
    /// Maximum absolute energy error observed so far
    pub max_abs_error: F,
    /// Mean absolute energy error
    pub mean_abs_error: F,
    /// Relative energy error |H(t) - H_0| / |H_0| (only if H_0 != 0)
    pub max_rel_error: F,
    /// Number of samples recorded
    sample_count: usize,
    /// Running sum of absolute errors
    error_sum: F,
}

impl<F: IntegrateFloat> EnergyMonitor<F> {
    /// Create a new energy monitor.
    pub fn new() -> Self {
        EnergyMonitor {
            initial_energy: None,
            energy_history: Vec::new(),
            abs_errors: Vec::new(),
            max_abs_error: F::zero(),
            mean_abs_error: F::zero(),
            max_rel_error: F::zero(),
            sample_count: 0,
            error_sum: F::zero(),
        }
    }

    /// Record an energy sample.
    pub fn record(&mut self, energy: F) {
        let h0 = match self.initial_energy {
            Some(h) => h,
            None => {
                self.initial_energy = Some(energy);
                self.energy_history.push(energy);
                self.abs_errors.push(F::zero());
                self.sample_count = 1;
                return;
            }
        };

        let abs_err = (energy - h0).abs();
        self.energy_history.push(energy);
        self.abs_errors.push(abs_err);
        self.sample_count += 1;
        self.error_sum += abs_err;

        if abs_err > self.max_abs_error {
            self.max_abs_error = abs_err;
        }

        let eps = F::from_f64(1e-300).unwrap_or_else(|| F::epsilon());
        if h0.abs() > eps {
            let rel_err = abs_err / h0.abs();
            if rel_err > self.max_rel_error {
                self.max_rel_error = rel_err;
            }
        }

        if self.sample_count > 0 {
            self.mean_abs_error =
                self.error_sum / F::from_usize(self.sample_count).unwrap_or_else(|| F::one());
        }
    }
}

impl<F: IntegrateFloat> Default for EnergyMonitor<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of symplectic ODE integration.
#[derive(Debug, Clone)]
pub struct SymplecticODEResult<F: IntegrateFloat> {
    /// Time points
    pub t: Vec<F>,
    /// Position coordinates at each time
    pub q: Vec<Array1<F>>,
    /// Momentum coordinates at each time
    pub p: Vec<Array1<F>>,
    /// Number of steps taken
    pub n_steps: usize,
    /// Number of function evaluations
    pub n_eval: usize,
    /// Energy monitoring data (present only if Hamiltonian was available)
    pub energy_monitor: Option<EnergyMonitor<F>>,
}

// ---------------------------------------------------------------------------
// Trait for symplectic steppers
// ---------------------------------------------------------------------------

/// Trait for symplectic one-step methods.
pub trait SymplecticStepper<F: IntegrateFloat> {
    /// Order of the method.
    fn order(&self) -> usize;

    /// Name for diagnostic output.
    fn name(&self) -> &str;

    /// Perform a single symplectic step.
    fn step(
        &self,
        sys: &dyn HamiltonianSystem<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)>;

    /// Integrate from t0 to tf with fixed step size dt.
    fn integrate(
        &self,
        sys: &dyn HamiltonianSystem<F>,
        t0: F,
        tf: F,
        dt: F,
        q0: Array1<F>,
        p0: Array1<F>,
    ) -> IntegrateResult<SymplecticODEResult<F>> {
        if dt <= F::zero() {
            return Err(IntegrateError::ValueError("dt must be positive".into()));
        }
        if q0.len() != p0.len() {
            return Err(IntegrateError::DimensionMismatch(
                "q and p must have the same length".into(),
            ));
        }

        let span = tf - t0;
        let n_steps_f = (span / dt).ceil();
        let n_steps = n_steps_f
            .to_f64()
            .ok_or_else(|| IntegrateError::ValueError("Cannot convert n_steps to f64".into()))?
            as usize;
        let actual_dt = span
            / F::from_usize(n_steps)
                .ok_or_else(|| IntegrateError::ValueError("Cannot convert n_steps".into()))?;

        let mut ts = Vec::with_capacity(n_steps + 1);
        let mut qs = Vec::with_capacity(n_steps + 1);
        let mut ps = Vec::with_capacity(n_steps + 1);

        ts.push(t0);
        qs.push(q0.clone());
        ps.push(p0.clone());

        let mut monitor = EnergyMonitor::new();
        let has_hamiltonian = sys.hamiltonian(t0, &q0, &p0).is_some();
        if let Some(h0) = sys.hamiltonian(t0, &q0, &p0) {
            monitor.record(h0);
        }

        let mut cur_t = t0;
        let mut cur_q = q0;
        let mut cur_p = p0;
        let mut n_eval: usize = 0;

        for _ in 0..n_steps {
            let (next_q, next_p) = self.step(sys, cur_t, &cur_q, &cur_p, actual_dt)?;
            // Approximate evals per step (depends on method, conservative estimate)
            n_eval += 2 * self.order();

            cur_t += actual_dt;
            if let Some(h) = sys.hamiltonian(cur_t, &next_q, &next_p) {
                monitor.record(h);
            }

            ts.push(cur_t);
            qs.push(next_q.clone());
            ps.push(next_p.clone());

            cur_q = next_q;
            cur_p = next_p;
        }

        Ok(SymplecticODEResult {
            t: ts,
            q: qs,
            p: ps,
            n_steps,
            n_eval,
            energy_monitor: if has_hamiltonian { Some(monitor) } else { None },
        })
    }
}

// ---------------------------------------------------------------------------
// Stormer-Verlet (Leapfrog) -- 2nd order
// ---------------------------------------------------------------------------

/// Stormer-Verlet (leapfrog) symplectic integrator, 2nd order.
///
/// Algorithm:
/// 1. p_{1/2} = p_n + (dt/2) dp/dt(t_n, q_n, p_n)
/// 2. q_{n+1} = q_n + dt  dq/dt(t_{n+1/2}, q_n, p_{1/2})
/// 3. p_{n+1} = p_{1/2} + (dt/2) dp/dt(t_{n+1}, q_{n+1}, p_{1/2})
#[derive(Debug, Clone)]
pub struct StormerVerletODE<F: IntegrateFloat> {
    _marker: PhantomData<F>,
}

impl<F: IntegrateFloat> StormerVerletODE<F> {
    pub fn new() -> Self {
        StormerVerletODE {
            _marker: PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for StormerVerletODE<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> SymplecticStepper<F> for StormerVerletODE<F> {
    fn order(&self) -> usize {
        2
    }
    fn name(&self) -> &str {
        "Stormer-Verlet"
    }

    fn step(
        &self,
        sys: &dyn HamiltonianSystem<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        let two = F::one() + F::one();
        let half_dt = dt / two;

        // Half-step momentum
        let dp1 = sys.dp_dt(t, q, p)?;
        let p_half = p + &(&dp1 * half_dt);

        // Full-step position
        let t_half = t + half_dt;
        let dq = sys.dq_dt(t_half, q, &p_half)?;
        let q_new = q + &(&dq * dt);

        // Half-step momentum
        let t_new = t + dt;
        let dp2 = sys.dp_dt(t_new, &q_new, &p_half)?;
        let p_new = &p_half + &(&dp2 * half_dt);

        Ok((q_new, p_new))
    }
}

// ---------------------------------------------------------------------------
// Velocity-Verlet -- 2nd order variant
// ---------------------------------------------------------------------------

/// Velocity Verlet symplectic integrator, 2nd order.
///
/// Equivalent to Stormer-Verlet for separable Hamiltonians but
/// formulated differently: updates position first using both
/// velocity and acceleration, then updates momentum.
#[derive(Debug, Clone)]
pub struct VelocityVerletODE<F: IntegrateFloat> {
    _marker: PhantomData<F>,
}

impl<F: IntegrateFloat> VelocityVerletODE<F> {
    pub fn new() -> Self {
        VelocityVerletODE {
            _marker: PhantomData,
        }
    }
}

impl<F: IntegrateFloat> Default for VelocityVerletODE<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> SymplecticStepper<F> for VelocityVerletODE<F> {
    fn order(&self) -> usize {
        2
    }
    fn name(&self) -> &str {
        "Velocity-Verlet"
    }

    fn step(
        &self,
        sys: &dyn HamiltonianSystem<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        let two = F::one() + F::one();
        let half_dt = dt / two;

        // Compute acceleration (force) at current state
        let dp_old = sys.dp_dt(t, q, p)?;

        // Half-step momentum
        let p_half = p + &(&dp_old * half_dt);

        // Full-step position using half-step momentum
        let dq = sys.dq_dt(t + half_dt, q, &p_half)?;
        let q_new = q + &(&dq * dt);

        // Compute acceleration at new position
        let t_new = t + dt;
        let dp_new = sys.dp_dt(t_new, &q_new, &p_half)?;

        // Complete momentum step
        let p_new = &p_half + &(&dp_new * half_dt);

        Ok((q_new, p_new))
    }
}

// ---------------------------------------------------------------------------
// Yoshida composition methods
// ---------------------------------------------------------------------------

/// Yoshida 4th order symplectic integrator.
///
/// Constructed by triple-jump composition of a 2nd-order base method
/// (Stormer-Verlet). Coefficients from Yoshida (1990):
///   w_1 = w_3 = 1/(2 - 2^{1/3})
///   w_0 = -2^{1/3}/(2 - 2^{1/3})
#[derive(Debug, Clone)]
pub struct Yoshida4<F: IntegrateFloat> {
    base: StormerVerletODE<F>,
    coefficients: [F; 3],
}

impl<F: IntegrateFloat> Yoshida4<F> {
    pub fn new() -> Self {
        let two = F::one() + F::one();
        let cbrt2 = two.powf(
            F::from_f64(1.0 / 3.0).unwrap_or_else(|| F::one() / (F::one() + F::one() + F::one())),
        );
        let w1 = F::one() / (two - cbrt2);
        let w0 = -cbrt2 / (two - cbrt2);

        Yoshida4 {
            base: StormerVerletODE::new(),
            coefficients: [w1, w0, w1],
        }
    }
}

impl<F: IntegrateFloat> Default for Yoshida4<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> SymplecticStepper<F> for Yoshida4<F> {
    fn order(&self) -> usize {
        4
    }
    fn name(&self) -> &str {
        "Yoshida-4"
    }

    fn step(
        &self,
        sys: &dyn HamiltonianSystem<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        let mut cur_t = t;
        let mut cur_q = q.clone();
        let mut cur_p = p.clone();

        for &c in &self.coefficients {
            let sub_dt = dt * c;
            let (nq, np) = self.base.step(sys, cur_t, &cur_q, &cur_p, sub_dt)?;
            cur_t += sub_dt;
            cur_q = nq;
            cur_p = np;
        }

        Ok((cur_q, cur_p))
    }
}

/// Yoshida 6th order symplectic integrator.
///
/// 7-stage composition with coefficients from Yoshida (1990).
#[derive(Debug, Clone)]
pub struct Yoshida6<F: IntegrateFloat> {
    base: StormerVerletODE<F>,
    coefficients: [F; 7],
}

impl<F: IntegrateFloat> Yoshida6<F> {
    pub fn new() -> Self {
        // Coefficients from Yoshida (1990) for 6th order
        let w1 = F::from_f64(0.784513610477560).unwrap_or_else(|| F::one());
        let w2 = F::from_f64(0.235573213359357).unwrap_or_else(|| F::one());
        let w3 = F::from_f64(-1.17767998417887).unwrap_or_else(|| -F::one());
        let w4 = F::from_f64(1.31518632068391).unwrap_or_else(|| F::one());

        Yoshida6 {
            base: StormerVerletODE::new(),
            coefficients: [w1, w2, w3, w4, w3, w2, w1],
        }
    }
}

impl<F: IntegrateFloat> Default for Yoshida6<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> SymplecticStepper<F> for Yoshida6<F> {
    fn order(&self) -> usize {
        6
    }
    fn name(&self) -> &str {
        "Yoshida-6"
    }

    fn step(
        &self,
        sys: &dyn HamiltonianSystem<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        let mut cur_t = t;
        let mut cur_q = q.clone();
        let mut cur_p = p.clone();

        for &c in &self.coefficients {
            let sub_dt = dt * c;
            let (nq, np) = self.base.step(sys, cur_t, &cur_q, &cur_p, sub_dt)?;
            cur_t += sub_dt;
            cur_q = nq;
            cur_p = np;
        }

        Ok((cur_q, cur_p))
    }
}

/// Yoshida 8th order symplectic integrator.
///
/// 15-stage composition with coefficients from Yoshida (1990).
#[derive(Debug, Clone)]
pub struct Yoshida8<F: IntegrateFloat> {
    base: StormerVerletODE<F>,
    coefficients: Vec<F>,
}

impl<F: IntegrateFloat> Yoshida8<F> {
    pub fn new() -> Self {
        // Coefficients for 8th-order Yoshida composition (Kahan & Li, 1997)
        let w = [
            F::from_f64(0.74167036435061).unwrap_or_else(|| F::one()),
            F::from_f64(-0.40910082580003).unwrap_or_else(|| -F::one()),
            F::from_f64(0.19075471029623).unwrap_or_else(|| F::one()),
            F::from_f64(-0.57386247111608).unwrap_or_else(|| -F::one()),
            F::from_f64(0.29906418130365).unwrap_or_else(|| F::one()),
            F::from_f64(0.33462491824529).unwrap_or_else(|| F::one()),
            F::from_f64(0.31529309239676).unwrap_or_else(|| F::one()),
            F::from_f64(-0.79688793935291).unwrap_or_else(|| -F::one()),
        ];

        // Symmetric composition: [w0..w7, w7..w0]
        let mut coefficients = Vec::with_capacity(15);
        for &c in &w {
            coefficients.push(c);
        }
        // Mirror from w[6] down to w[0]
        for i in (0..7).rev() {
            coefficients.push(w[i]);
        }

        Yoshida8 {
            base: StormerVerletODE::new(),
            coefficients,
        }
    }
}

impl<F: IntegrateFloat> Default for Yoshida8<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: IntegrateFloat> SymplecticStepper<F> for Yoshida8<F> {
    fn order(&self) -> usize {
        8
    }
    fn name(&self) -> &str {
        "Yoshida-8"
    }

    fn step(
        &self,
        sys: &dyn HamiltonianSystem<F>,
        t: F,
        q: &Array1<F>,
        p: &Array1<F>,
        dt: F,
    ) -> IntegrateResult<(Array1<F>, Array1<F>)> {
        let mut cur_t = t;
        let mut cur_q = q.clone();
        let mut cur_p = p.clone();

        for &c in &self.coefficients {
            let sub_dt = dt * c;
            let (nq, np) = self.base.step(sys, cur_t, &cur_q, &cur_p, sub_dt)?;
            cur_t += sub_dt;
            cur_q = nq;
            cur_p = np;
        }

        Ok((cur_q, cur_p))
    }
}

// ---------------------------------------------------------------------------
// Convenience: solve_hamiltonian
// ---------------------------------------------------------------------------

/// Solve a Hamiltonian system using a specified symplectic method.
///
/// # Arguments
/// * `sys` - the Hamiltonian system
/// * `method` - the symplectic stepper
/// * `t0` - initial time
/// * `tf` - final time
/// * `dt` - step size
/// * `q0` - initial positions
/// * `p0` - initial momenta
pub fn solve_hamiltonian<F: IntegrateFloat>(
    sys: &dyn HamiltonianSystem<F>,
    method: &dyn SymplecticStepper<F>,
    t0: F,
    tf: F,
    dt: F,
    q0: Array1<F>,
    p0: Array1<F>,
) -> IntegrateResult<SymplecticODEResult<F>> {
    method.integrate(sys, t0, tf, dt, q0, p0)
}

/// Enumeration of available symplectic methods for convenience.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymplecticMethod {
    /// Stormer-Verlet (leapfrog), order 2
    StormerVerlet,
    /// Velocity Verlet, order 2
    VelocityVerlet,
    /// Yoshida 4th order
    Yoshida4,
    /// Yoshida 6th order
    Yoshida6,
    /// Yoshida 8th order
    Yoshida8,
}

/// Create a boxed stepper from the method enum.
pub fn create_stepper<F: IntegrateFloat>(
    method: SymplecticMethod,
) -> Box<dyn SymplecticStepper<F>> {
    match method {
        SymplecticMethod::StormerVerlet => Box::new(StormerVerletODE::<F>::new()),
        SymplecticMethod::VelocityVerlet => Box::new(VelocityVerletODE::<F>::new()),
        SymplecticMethod::Yoshida4 => Box::new(Yoshida4::<F>::new()),
        SymplecticMethod::Yoshida6 => Box::new(Yoshida6::<F>::new()),
        SymplecticMethod::Yoshida8 => Box::new(Yoshida8::<F>::new()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Build a simple harmonic oscillator: H = p^2/2 + q^2/2
    fn harmonic_oscillator() -> SeparableSystem<f64> {
        SeparableSystem::new(
            1,
            |_t, p: &Array1<f64>| p.clone(), // dT/dp = p
            |_t, q: &Array1<f64>| q.clone(), // dV/dq = q
        )
        .with_energy(
            |_t, p: &Array1<f64>| 0.5 * p.dot(p), // T(p) = p^2/2
            |_t, q: &Array1<f64>| 0.5 * q.dot(q), // V(q) = q^2/2
        )
    }

    /// Build a 2D Kepler problem: H = |p|^2/2 - 1/|q|
    fn kepler_2d() -> SeparableSystem<f64> {
        SeparableSystem::new(
            2,
            |_t, p: &Array1<f64>| p.clone(),
            |_t, q: &Array1<f64>| {
                let r2 = q[0] * q[0] + q[1] * q[1];
                let r = r2.sqrt();
                if r < 1e-12 {
                    Array1::zeros(2)
                } else {
                    // dV/dq = -d/dq(-1/r) = q/r^3
                    let r3 = r * r2;
                    array![q[0] / r3, q[1] / r3]
                }
            },
        )
        .with_energy(
            |_t, p: &Array1<f64>| 0.5 * p.dot(p),
            |_t, q: &Array1<f64>| {
                let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
                if r < 1e-12 {
                    0.0
                } else {
                    -1.0 / r
                }
            },
        )
    }

    #[test]
    fn test_stormer_verlet_harmonic() {
        let sys = harmonic_oscillator();
        let sv = StormerVerletODE::new();
        let q0 = array![1.0_f64];
        let p0 = array![0.0_f64];

        let result = sv
            .integrate(
                &sys,
                0.0,
                2.0 * std::f64::consts::PI,
                0.01,
                q0.clone(),
                p0.clone(),
            )
            .expect("integration should succeed");

        // After one full period, should return close to initial state
        let q_final = result.q.last().expect("should have final q");
        let p_final = result.p.last().expect("should have final p");
        assert!(
            (q_final[0] - 1.0).abs() < 0.01,
            "q should return near 1.0, got {}",
            q_final[0]
        );
        assert!(
            p_final[0].abs() < 0.01,
            "p should return near 0.0, got {}",
            p_final[0]
        );

        // Energy conservation
        let mon = result.energy_monitor.as_ref().expect("should have monitor");
        // Stormer-Verlet is 2nd order: energy error bounded by O(dt^2) ~ 1e-4 for dt=0.01
        assert!(
            mon.max_rel_error < 1e-3,
            "energy drift too large: {}",
            mon.max_rel_error
        );
    }

    #[test]
    fn test_velocity_verlet_harmonic() {
        let sys = harmonic_oscillator();
        let vv = VelocityVerletODE::new();
        let q0 = array![1.0_f64];
        let p0 = array![0.0_f64];

        let result = vv
            .integrate(&sys, 0.0, 2.0 * std::f64::consts::PI, 0.01, q0, p0)
            .expect("integration should succeed");

        let mon = result.energy_monitor.as_ref().expect("should have monitor");
        // Velocity-Verlet is 2nd order: energy error bounded by O(dt^2) ~ 1e-4 for dt=0.01
        assert!(
            mon.max_rel_error < 1e-3,
            "energy drift too large: {}",
            mon.max_rel_error
        );
    }

    #[test]
    fn test_yoshida4_convergence() {
        let sys = harmonic_oscillator();
        let y4 = Yoshida4::new();
        let sv = StormerVerletODE::new();

        let q0 = array![1.0_f64];
        let p0 = array![0.0_f64];
        let tf = 1.0;

        // Exact solution at t=1: q = cos(1), p = -sin(1)
        let q_exact = 1.0_f64.cos();
        let p_exact = -1.0_f64.sin();

        // Compare errors at two step sizes to verify order
        let dts = [0.1, 0.05];
        let mut sv_errors = Vec::new();
        let mut y4_errors = Vec::new();

        for &dt in &dts {
            let sv_res = sv
                .integrate(&sys, 0.0, tf, dt, q0.clone(), p0.clone())
                .expect("sv integration failed");
            let y4_res = y4
                .integrate(&sys, 0.0, tf, dt, q0.clone(), p0.clone())
                .expect("y4 integration failed");

            let sv_err = ((sv_res.q.last().expect("no q")[0] - q_exact).powi(2)
                + (sv_res.p.last().expect("no p")[0] - p_exact).powi(2))
            .sqrt();
            let y4_err = ((y4_res.q.last().expect("no q")[0] - q_exact).powi(2)
                + (y4_res.p.last().expect("no p")[0] - p_exact).powi(2))
            .sqrt();

            sv_errors.push(sv_err);
            y4_errors.push(y4_err);
        }

        // When dt halved, 2nd-order error should decrease by ~4, 4th-order by ~16
        let sv_ratio = sv_errors[0] / sv_errors[1];
        let y4_ratio = y4_errors[0] / y4_errors[1];

        assert!(
            sv_ratio > 3.0 && sv_ratio < 5.0,
            "SV convergence ratio {sv_ratio} not ~4"
        );
        assert!(
            y4_ratio > 12.0 && y4_ratio < 20.0,
            "Y4 convergence ratio {y4_ratio} not ~16"
        );

        // Y4 should be more accurate than SV at same step size
        assert!(y4_errors[0] < sv_errors[0], "Y4 should beat SV accuracy");
    }

    #[test]
    fn test_yoshida6_better_than_yoshida4() {
        let sys = harmonic_oscillator();
        let y4 = Yoshida4::new();
        let y6 = Yoshida6::new();

        let q0 = array![1.0_f64];
        let p0 = array![0.0_f64];
        let dt = 0.1;
        let tf = 1.0;

        let q_exact = 1.0_f64.cos();
        let p_exact = -1.0_f64.sin();

        let r4 = y4
            .integrate(&sys, 0.0, tf, dt, q0.clone(), p0.clone())
            .expect("y4 failed");
        let r6 = y6
            .integrate(&sys, 0.0, tf, dt, q0.clone(), p0.clone())
            .expect("y6 failed");

        let e4 = ((r4.q.last().expect("no q")[0] - q_exact).powi(2)
            + (r4.p.last().expect("no p")[0] - p_exact).powi(2))
        .sqrt();
        let e6 = ((r6.q.last().expect("no q")[0] - q_exact).powi(2)
            + (r6.p.last().expect("no p")[0] - p_exact).powi(2))
        .sqrt();

        assert!(
            e6 < e4,
            "Y6 error ({e6}) should be less than Y4 error ({e4})"
        );
    }

    #[test]
    fn test_yoshida8_high_accuracy() {
        let sys = harmonic_oscillator();
        let y8 = Yoshida8::new();

        let q0 = array![1.0_f64];
        let p0 = array![0.0_f64];
        let dt = 0.1;
        let tf = 1.0;

        let q_exact = 1.0_f64.cos();
        let p_exact = -1.0_f64.sin();

        let r8 = y8.integrate(&sys, 0.0, tf, dt, q0, p0).expect("y8 failed");

        let e8 = ((r8.q.last().expect("no q")[0] - q_exact).powi(2)
            + (r8.p.last().expect("no p")[0] - p_exact).powi(2))
        .sqrt();

        // 8th order with dt=0.1 should be very accurate
        assert!(e8 < 1e-8, "Y8 error {e8} too large");
    }

    #[test]
    fn test_kepler_energy_conservation() {
        let sys = kepler_2d();
        let y4 = Yoshida4::new();

        // Circular orbit
        let q0 = array![1.0, 0.0];
        let p0 = array![0.0, 1.0];

        let result = y4
            .integrate(&sys, 0.0, 20.0, 0.01, q0, p0)
            .expect("kepler integration failed");

        let mon = result.energy_monitor.as_ref().expect("should have monitor");
        assert!(
            mon.max_rel_error < 1e-6,
            "Kepler energy drift too large: {}",
            mon.max_rel_error
        );
    }

    #[test]
    fn test_energy_monitor_recording() {
        let sys = harmonic_oscillator();
        let sv = StormerVerletODE::new();
        let q0 = array![1.0_f64];
        let p0 = array![0.0_f64];

        let result = sv
            .integrate(&sys, 0.0, 1.0, 0.1, q0, p0)
            .expect("integration failed");

        let mon = result.energy_monitor.as_ref().expect("should have monitor");
        assert!(mon.initial_energy.is_some());
        assert!(!mon.energy_history.is_empty());
        assert_eq!(mon.energy_history.len(), mon.abs_errors.len());

        // Initial energy should be 0.5 for q=1,p=0
        let h0 = mon.initial_energy.expect("should have initial energy");
        assert!((h0 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_create_stepper() {
        let stepper = create_stepper::<f64>(SymplecticMethod::Yoshida4);
        assert_eq!(stepper.order(), 4);
        assert_eq!(stepper.name(), "Yoshida-4");

        let stepper = create_stepper::<f64>(SymplecticMethod::Yoshida8);
        assert_eq!(stepper.order(), 8);
    }

    #[test]
    fn test_solve_hamiltonian_convenience() {
        let sys = harmonic_oscillator();
        let stepper = Yoshida4::<f64>::new();

        let result = solve_hamiltonian(
            &sys,
            &stepper,
            0.0,
            std::f64::consts::PI,
            0.01,
            array![1.0],
            array![0.0],
        )
        .expect("solve_hamiltonian failed");

        // At t=pi, q should be cos(pi) = -1, p should be -sin(pi) ~ 0
        let q_f = result.q.last().expect("no q");
        let p_f = result.p.last().expect("no p");
        assert!((q_f[0] + 1.0).abs() < 0.01, "q should be near -1");
        assert!(p_f[0].abs() < 0.01, "p should be near 0");
    }

    #[test]
    fn test_invalid_inputs() {
        let sys = harmonic_oscillator();
        let sv = StormerVerletODE::new();

        // Negative dt
        let res = sv.integrate(&sys, 0.0, 1.0, -0.1, array![1.0], array![0.0]);
        assert!(res.is_err());

        // Mismatched dimensions
        let res = sv.integrate(&sys, 0.0, 1.0, 0.1, array![1.0, 2.0], array![0.0]);
        assert!(res.is_err());
    }

    #[test]
    fn test_long_time_energy_bounded() {
        // Symplectic integrators should have bounded energy error, not growing
        let sys = harmonic_oscillator();
        let y4 = Yoshida4::new();

        let result = y4
            .integrate(&sys, 0.0, 100.0, 0.05, array![1.0], array![0.0])
            .expect("long integration failed");

        let mon = result.energy_monitor.as_ref().expect("monitor");
        // Energy error should stay bounded even after long integration
        assert!(
            mon.max_abs_error < 1e-6,
            "Energy error grew too large over long time: {}",
            mon.max_abs_error
        );
    }
}
