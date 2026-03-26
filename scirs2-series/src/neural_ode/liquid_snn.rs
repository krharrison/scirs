//! Liquid State Networks and Liquid Time-Constant (LTC) Networks
//!
//! ## Liquid Time-Constant (LTC) Neuron
//!
//! The LTC neuron (Hasani et al., 2021) solves:
//!
//! ```text
//!   τ(x, t) · dx/dt = -x + f(W_rec x + W_in u + bias)
//! ```
//!
//! where the **time constant is input-dependent**:
//!
//! ```text
//!   τ(x, u) = τ_0 + A σ(W_τ x + W_τu u + b_τ)
//! ```
//!
//! This creates *adaptive* dynamics: the network can speed up or slow down its
//! time constant based on the current input, giving it richer expressivity than
//! a plain ODE-RNN.
//!
//! ## Closed-Form Continuous-Depth (CfC) Approximation
//!
//! A closed-form approximation avoids expensive ODE solving at inference time
//! (Hasani et al., 2022):
//!
//! ```text
//!   x(t + Δt) ≈ σ(-g(x,u)·Δt) · x(t)  +  (1 - σ(-g(x,u)·Δt)) · f(x,u)
//! ```
//!
//! where `g(x,u)` is a learned gating scalar and `f(x,u)` is the steady-state
//! solution.  This is implemented in [`CfCCell`].
//!
//! ## Liquid State Network (LSN)
//!
//! A recurrent reservoir (echo-state network style) of LTC neurons, with a
//! fixed or randomly-initialised sparse reservoir and a trainable linear readout.

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Utility activations
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid<F: Float + FromPrimitive>(x: F) -> F {
    let one = F::from(1.0).expect("1.0");
    one / (one + (-x).exp())
}

#[inline]
fn tanh_act<F: Float>(x: F) -> F {
    x.tanh()
}

/// Apply sigmoid element-wise.
fn sigmoid_vec<F: Float + FromPrimitive>(v: &Array1<F>) -> Array1<F> {
    let n = v.len();
    let mut out = Array1::zeros(n);
    for i in 0..n {
        out[i] = sigmoid(v[i]);
    }
    out
}

/// Dense layer: `y = W x + b` (no activation).
fn linear<F: Float + FromPrimitive>(w: &Array2<F>, b: &Array1<F>, x: &Array1<F>) -> Array1<F> {
    let r = w.nrows();
    let c = w.ncols();
    let mut y = Array1::zeros(r);
    for i in 0..r {
        let mut s = b[i];
        for j in 0..c {
            s = s + w[[i, j]] * x[j];
        }
        y[i] = s;
    }
    y
}

/// Dense layer with tanh: `y = tanh(W x + b)`.
fn linear_tanh<F: Float + FromPrimitive>(w: &Array2<F>, b: &Array1<F>, x: &Array1<F>) -> Array1<F> {
    let mut y = linear(w, b, x);
    for v in y.iter_mut() {
        *v = tanh_act(*v);
    }
    y
}

// ---------------------------------------------------------------------------
// Weight initialisation helper
// ---------------------------------------------------------------------------

fn random_matrix<F: Float + FromPrimitive>(
    rows: usize,
    cols: usize,
    std_dev: F,
    seed: u64,
) -> Array2<F> {
    let mut mat = Array2::zeros((rows, cols));
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in 0..rows {
        for j in 0..cols {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let fval = (state as f64) / (u64::MAX as f64) - 0.5;
            mat[[i, j]] = F::from(fval * 2.0).expect("rand") * std_dev;
        }
    }
    mat
}

fn random_vec<F: Float + FromPrimitive>(len: usize, std_dev: F, seed: u64) -> Array1<F> {
    let mut v = Array1::zeros(len);
    let mut state = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    for i in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let fval = (state as f64) / (u64::MAX as f64) - 0.5;
        v[i] = F::from(fval * 2.0).expect("rand") * std_dev;
    }
    v
}

/// Create a sparse binary adjacency matrix (Erdős–Rényi style) for the reservoir.
fn sparse_adjacency(size: usize, sparsity: f64, seed: u64) -> Array2<f64> {
    let mut mat = Array2::<f64>::zeros((size, size));
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    for i in 0..size {
        for j in 0..size {
            if i == j {
                continue;
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let p = (state as f64) / (u64::MAX as f64);
            if p < sparsity {
                let sign_state = state.wrapping_add(1);
                mat[[i, j]] = if sign_state & 1 == 0 { 1.0 } else { -1.0 };
            }
        }
    }
    mat
}

// ---------------------------------------------------------------------------
// LTC neuron (single cell)
// ---------------------------------------------------------------------------

/// State of an LTC cell.
#[derive(Debug, Clone)]
pub struct LtcState<F: Float + Clone> {
    /// Cell activation vector `x ∈ ℝ^{hidden_dim}`
    pub x: Array1<F>,
}

impl<F: Float + Clone + FromPrimitive> LtcState<F> {
    /// Zero-initialised state.
    pub fn zeros(hidden_dim: usize) -> Self {
        Self {
            x: Array1::zeros(hidden_dim),
        }
    }
}

/// A layer of Liquid Time-Constant neurons.
///
/// The dynamics are:
/// ```text
///   τ(x, u) = τ₀ + A · σ(W_τ [x; u] + b_τ)     (scalar per neuron)
///   f(x, u)  = tanh(W_rec x + W_in u + b_f)
///   dx/dt    = (-x + f(x,u)) / τ(x,u)
/// ```
///
/// Integration uses Euler's method with a fixed step size `dt`.
#[derive(Debug, Clone)]
pub struct LtcCell<F: Float + Debug + FromPrimitive + Clone> {
    hidden_dim: usize,
    input_dim: usize,
    /// Recurrent weight matrix `[hidden × hidden]`
    w_rec: Array2<F>,
    /// Input weight matrix `[hidden × input]`
    w_in: Array2<F>,
    /// Bias for the nonlinearity `[hidden]`
    b_f: Array1<F>,
    /// Time-constant gate weights `[hidden × (hidden + input)]`
    w_tau: Array2<F>,
    b_tau: Array1<F>,
    /// Base time constant τ₀ (scalar, > 0)
    tau0: F,
    /// Amplitude of adaptive time constant A (scalar, > 0)
    tau_amplitude: F,
}

impl<F: Float + Debug + FromPrimitive + Clone> LtcCell<F> {
    /// Construct an LTC cell with random initialisation.
    ///
    /// # Arguments
    /// * `input_dim` – number of input features.
    /// * `hidden_dim` – number of LTC neurons.
    /// * `tau0` – base time constant (positive; e.g. `1.0`).
    /// * `seed` – RNG seed.
    pub fn new(input_dim: usize, hidden_dim: usize, tau0: F, seed: u64) -> Self {
        let concat_dim = hidden_dim + input_dim;
        let std_rec = F::from((2.0 / (hidden_dim + hidden_dim) as f64).sqrt()).expect("std");
        let std_in = F::from((2.0 / (input_dim + hidden_dim) as f64).sqrt()).expect("std");
        let std_tau = F::from((1.0 / concat_dim as f64).sqrt()).expect("std");

        Self {
            hidden_dim,
            input_dim,
            w_rec: random_matrix(hidden_dim, hidden_dim, std_rec, seed),
            w_in: random_matrix(hidden_dim, input_dim, std_in, seed.wrapping_add(1)),
            b_f: Array1::zeros(hidden_dim),
            w_tau: random_matrix(hidden_dim, concat_dim, std_tau, seed.wrapping_add(2)),
            b_tau: Array1::zeros(hidden_dim),
            tau0,
            tau_amplitude: F::from(1.0).expect("1.0"),
        }
    }

    /// Compute the adaptive time constant vector `τ(x, u) ∈ ℝ^{hidden_dim}`.
    pub fn time_constants(&self, x: &Array1<F>, u: &Array1<F>) -> Array1<F> {
        // Concatenate [x; u]
        let concat_dim = self.hidden_dim + self.input_dim;
        let mut xu = Array1::zeros(concat_dim);
        for i in 0..self.hidden_dim {
            xu[i] = x[i];
        }
        for i in 0..self.input_dim {
            xu[self.hidden_dim + i] = u[i];
        }
        let gate = linear(&self.w_tau, &self.b_tau, &xu);
        let sig = sigmoid_vec(&gate);
        let mut tau = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            tau[i] = self.tau0 + self.tau_amplitude * sig[i];
        }
        tau
    }

    /// One Euler integration step: `x ← x + dt * (-x + f(x,u)) / τ(x,u)`.
    ///
    /// Returns the new state.
    pub fn step(&self, state: &LtcState<F>, u: &Array1<F>, dt: F) -> Result<LtcState<F>> {
        if u.len() != self.input_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.input_dim,
                actual: u.len(),
            });
        }
        if state.x.len() != self.hidden_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.hidden_dim,
                actual: state.x.len(),
            });
        }

        // f(x, u) = tanh(W_rec x + W_in u + b_f)
        let rec_contrib = linear(&self.w_rec, &self.b_f, &state.x);
        let in_contrib = {
            let raw = linear(&self.w_in, &Array1::zeros(self.hidden_dim), u);
            // Add element-wise: rec_contrib already has b_f; just add in_contrib
            let mut combined = Array1::zeros(self.hidden_dim);
            for i in 0..self.hidden_dim {
                combined[i] = rec_contrib[i] + raw[i];
            }
            combined
        };
        let f_xu: Array1<F> = in_contrib
            .iter()
            .map(|v| tanh_act(*v))
            .collect::<Vec<_>>()
            .into();

        let tau = self.time_constants(&state.x, u);

        let mut x_new = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            let dx = (f_xu[i] - state.x[i]) / tau[i];
            x_new[i] = state.x[i] + dt * dx;
        }
        Ok(LtcState { x: x_new })
    }

    /// Process a sequence of inputs with a given time-step `dt`.
    ///
    /// Returns the sequence of hidden states (one per input).
    pub fn forward_sequence(&self, inputs: &[Array1<F>], dt: F) -> Result<Vec<LtcState<F>>> {
        let mut state = LtcState::zeros(self.hidden_dim);
        let mut states = Vec::with_capacity(inputs.len());
        for u in inputs {
            state = self.step(&state, u, dt)?;
            states.push(state.clone());
        }
        Ok(states)
    }

    /// Process an irregularly-sampled sequence.
    ///
    /// # Arguments
    /// * `times` – timestamps (monotonically non-decreasing).
    /// * `inputs` – input vectors at those timestamps.
    pub fn forward_irregular(&self, times: &[F], inputs: &[Array1<F>]) -> Result<Vec<LtcState<F>>> {
        if times.len() != inputs.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: times.len(),
                actual: inputs.len(),
            });
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut state = LtcState::zeros(self.hidden_dim);
        let mut states = Vec::with_capacity(inputs.len());

        // First step uses a default dt of 1.0
        state = self.step(&state, &inputs[0], F::one())?;
        states.push(state.clone());

        for i in 1..times.len() {
            let dt = (times[i] - times[i - 1]).abs().max(F::epsilon());
            state = self.step(&state, &inputs[i], dt)?;
            states.push(state.clone());
        }
        Ok(states)
    }

    /// Return the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Return the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
}

// ---------------------------------------------------------------------------
// Closed-Form Continuous-depth (CfC) cell
// ---------------------------------------------------------------------------

/// Closed-Form Continuous-depth (CfC) cell.
///
/// Approximates the LTC ODE with:
/// ```text
///   g   = σ(W_g [x; u] + b_g)              (gating, same shape as x)
///   f   = tanh(W_f [x; u] + b_f)            (steady-state attractor)
///   x'  = g ⊙ x  +  (1 − g) ⊙ f           (interpolation, Δt absorbed into g)
/// ```
///
/// The time step `Δt` modulates the gate: `g_t = σ(W_g [x;u] − Δt · b_t + b_g)`
/// where `b_t` is a learned time-bias vector, so the cell is aware of elapsed time.
#[derive(Debug, Clone)]
pub struct CfCCell<F: Float + Debug + FromPrimitive + Clone> {
    hidden_dim: usize,
    input_dim: usize,
    concat_dim: usize,
    /// Gate weights `[hidden × concat]`
    w_g: Array2<F>,
    b_g: Array1<F>,
    /// Steady-state weights `[hidden × concat]`
    w_f: Array2<F>,
    b_f: Array1<F>,
    /// Time bias `[hidden]` (per-neuron, multiplies Δt before sigmoid)
    b_t: Array1<F>,
}

impl<F: Float + Debug + FromPrimitive + Clone> CfCCell<F> {
    /// Construct a CfC cell.
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let concat_dim = hidden_dim + input_dim;
        let std = F::from((2.0 / (concat_dim + hidden_dim) as f64).sqrt()).expect("std");
        Self {
            hidden_dim,
            input_dim,
            concat_dim,
            w_g: random_matrix(hidden_dim, concat_dim, std, seed),
            b_g: Array1::zeros(hidden_dim),
            w_f: random_matrix(hidden_dim, concat_dim, std, seed.wrapping_add(1)),
            b_f: Array1::zeros(hidden_dim),
            b_t: random_vec(hidden_dim, F::from(0.1).expect("0.1"), seed.wrapping_add(2)),
        }
    }

    /// Single CfC step with explicit time delta `dt`.
    pub fn step(&self, x: &Array1<F>, u: &Array1<F>, dt: F) -> Result<Array1<F>> {
        if u.len() != self.input_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.input_dim,
                actual: u.len(),
            });
        }
        if x.len() != self.hidden_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.hidden_dim,
                actual: x.len(),
            });
        }

        // Concatenate [x; u]
        let mut xu = Array1::zeros(self.concat_dim);
        for i in 0..self.hidden_dim {
            xu[i] = x[i];
        }
        for i in 0..self.input_dim {
            xu[self.hidden_dim + i] = u[i];
        }

        // Gate: σ(W_g [x;u] - dt * b_t + b_g)
        let gate_logit = {
            let raw = linear(&self.w_g, &self.b_g, &xu);
            let mut out = Array1::zeros(self.hidden_dim);
            for i in 0..self.hidden_dim {
                out[i] = sigmoid(raw[i] - dt * self.b_t[i]);
            }
            out
        };

        // Steady state: tanh(W_f [x;u] + b_f)
        let f_val = linear_tanh(&self.w_f, &self.b_f, &xu);

        // Interpolate
        let mut x_new = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            x_new[i] = gate_logit[i] * x[i] + (F::one() - gate_logit[i]) * f_val[i];
        }
        Ok(x_new)
    }

    /// Process a sequence with uniform time step.
    pub fn forward_sequence(&self, inputs: &[Array1<F>], dt: F) -> Result<Vec<Array1<F>>> {
        let mut x = Array1::zeros(self.hidden_dim);
        let mut out = Vec::with_capacity(inputs.len());
        for u in inputs {
            x = self.step(&x, u, dt)?;
            out.push(x.clone());
        }
        Ok(out)
    }

    /// Process an irregularly-sampled sequence.
    pub fn forward_irregular(&self, times: &[F], inputs: &[Array1<F>]) -> Result<Vec<Array1<F>>> {
        if times.len() != inputs.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: times.len(),
                actual: inputs.len(),
            });
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut x = Array1::zeros(self.hidden_dim);
        let mut out = Vec::with_capacity(inputs.len());

        x = self.step(&x, &inputs[0], F::one())?;
        out.push(x.clone());

        for i in 1..times.len() {
            let dt = (times[i] - times[i - 1]).abs().max(F::epsilon());
            x = self.step(&x, &inputs[i], dt)?;
            out.push(x.clone());
        }
        Ok(out)
    }

    /// Return hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Return input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
}

// ---------------------------------------------------------------------------
// Liquid State Network (reservoir + readout)
// ---------------------------------------------------------------------------

/// Configuration for a Liquid State Network.
#[derive(Debug, Clone)]
pub struct LiquidSNNConfig {
    /// Input dimension.
    pub input_dim: usize,
    /// Number of reservoir neurons.
    pub reservoir_size: usize,
    /// Sparsity of reservoir connections (fraction of non-zero weights).
    pub reservoir_sparsity: f64,
    /// Output / readout dimension.
    pub output_dim: usize,
    /// Euler integration step size.
    pub dt: f64,
    /// Base time constant τ₀.
    pub tau0: f64,
    /// Use CfC cells instead of LTC cells (faster, closed-form).
    pub use_cfc: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for LiquidSNNConfig {
    fn default() -> Self {
        Self {
            input_dim: 1,
            reservoir_size: 32,
            reservoir_sparsity: 0.1,
            output_dim: 1,
            dt: 0.1,
            tau0: 1.0,
            use_cfc: false,
            seed: 42,
        }
    }
}

/// Liquid State Network with an LTC or CfC reservoir and a linear readout.
///
/// The reservoir weights are fixed at initialisation (echo-state style).
/// Only the linear readout `W_out` is intended to be trained.
#[derive(Debug, Clone)]
pub struct LiquidSNN<F: Float + Debug + FromPrimitive + Clone> {
    config: LiquidSNNConfig,
    /// LTC cell (used when `use_cfc == false`)
    ltc_cell: Option<LtcCell<F>>,
    /// CfC cell (used when `use_cfc == true`)
    cfc_cell: Option<CfCCell<F>>,
    /// Linear readout: `[output_dim × reservoir_size]`
    readout_w: Array2<F>,
    readout_b: Array1<F>,
}

impl<F: Float + Debug + FromPrimitive + Clone> LiquidSNN<F> {
    /// Build a Liquid State Network.
    pub fn new(config: LiquidSNNConfig) -> Result<Self> {
        if config.reservoir_size == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "reservoir_size".into(),
                message: "must be ≥ 1".into(),
            });
        }
        let s = config.seed;
        let tau0 = F::from(config.tau0).expect("tau0");
        let dt_f = config.dt;

        let (ltc_cell, cfc_cell) = if config.use_cfc {
            let cell = CfCCell::new(config.input_dim, config.reservoir_size, s);
            (None, Some(cell))
        } else {
            let cell = LtcCell::new(config.input_dim, config.reservoir_size, tau0, s);
            (Some(cell), None)
        };

        let std_out = F::from((2.0 / (config.reservoir_size + config.output_dim) as f64).sqrt())
            .expect("std");
        let readout_w = random_matrix(
            config.output_dim,
            config.reservoir_size,
            std_out,
            s.wrapping_add(99),
        );
        let readout_b = Array1::zeros(config.output_dim);

        let _ = dt_f; // will be used during inference

        Ok(Self {
            config,
            ltc_cell,
            cfc_cell,
            readout_w,
            readout_b,
        })
    }

    /// Run the reservoir on a uniformly-sampled input sequence.
    ///
    /// Returns reservoir state trajectories for each input step.
    pub fn run_reservoir(&self, inputs: &[Array1<F>]) -> Result<Vec<Array1<F>>> {
        let dt = F::from(self.config.dt).expect("dt");
        if let Some(ltc) = &self.ltc_cell {
            let states = ltc.forward_sequence(inputs, dt)?;
            Ok(states.into_iter().map(|s| s.x).collect())
        } else if let Some(cfc) = &self.cfc_cell {
            cfc.forward_sequence(inputs, dt)
        } else {
            Err(TimeSeriesError::InvalidModel("no cell available".into()))
        }
    }

    /// Run on an irregularly-sampled sequence.
    pub fn run_reservoir_irregular(
        &self,
        times: &[F],
        inputs: &[Array1<F>],
    ) -> Result<Vec<Array1<F>>> {
        if let Some(ltc) = &self.ltc_cell {
            let states = ltc.forward_irregular(times, inputs)?;
            Ok(states.into_iter().map(|s| s.x).collect())
        } else if let Some(cfc) = &self.cfc_cell {
            cfc.forward_irregular(times, inputs)
        } else {
            Err(TimeSeriesError::InvalidModel("no cell available".into()))
        }
    }

    /// Apply the linear readout to a single reservoir state.
    pub fn readout(&self, reservoir_state: &Array1<F>) -> Result<Array1<F>> {
        if reservoir_state.len() != self.config.reservoir_size {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.reservoir_size,
                actual: reservoir_state.len(),
            });
        }
        let d = self.config.output_dim;
        let mut out = Array1::zeros(d);
        for i in 0..d {
            let mut s = self.readout_b[i];
            for j in 0..self.config.reservoir_size {
                s = s + self.readout_w[[i, j]] * reservoir_state[j];
            }
            out[i] = s;
        }
        Ok(out)
    }

    /// Full forward pass: reservoir → readout for each time step.
    pub fn forward(&self, inputs: &[Array1<F>]) -> Result<Vec<Array1<F>>> {
        let reservoir_states = self.run_reservoir(inputs)?;
        reservoir_states.iter().map(|s| self.readout(s)).collect()
    }

    /// Return config reference.
    pub fn config(&self) -> &LiquidSNNConfig {
        &self.config
    }

    /// Return the LTC cell (if in use).
    pub fn ltc_cell(&self) -> Option<&LtcCell<F>> {
        self.ltc_cell.as_ref()
    }

    /// Return the CfC cell (if in use).
    pub fn cfc_cell(&self) -> Option<&CfCCell<F>> {
        self.cfc_cell.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ltc_step_shape() {
        let cell = LtcCell::<f64>::new(3, 8, 1.0, 0);
        let state = LtcState::zeros(8);
        let u = array![0.1_f64, -0.2, 0.3];
        let new_state = cell.step(&state, &u, 0.1).expect("step");
        assert_eq!(new_state.x.len(), 8);
    }

    #[test]
    fn test_ltc_time_constants_positive() {
        let cell = LtcCell::<f64>::new(2, 4, 1.0, 1);
        let x = array![0.0_f64, 0.5, -0.5, 1.0];
        let u = array![0.1_f64, 0.2];
        let tau = cell.time_constants(&x, &u);
        for &t in tau.iter() {
            assert!(t > 0.0, "time constant must be positive, got {t}");
        }
    }

    #[test]
    fn test_ltc_sequence() {
        let cell = LtcCell::<f64>::new(1, 4, 0.5, 2);
        let inputs: Vec<Array1<f64>> = (0..10).map(|i| array![(i as f64) * 0.1]).collect();
        let states = cell.forward_sequence(&inputs, 0.05).expect("seq");
        assert_eq!(states.len(), 10);
    }

    #[test]
    fn test_cfc_step_shape() {
        let cell = CfCCell::<f64>::new(2, 6, 5);
        let x = Array1::zeros(6);
        let u = array![0.5_f64, -0.3];
        let x_new = cell.step(&x, &u, 0.1).expect("cfc step");
        assert_eq!(x_new.len(), 6);
    }

    #[test]
    fn test_liquid_snn_ltc() {
        let snn = LiquidSNN::<f64>::new(LiquidSNNConfig {
            input_dim: 2,
            reservoir_size: 8,
            output_dim: 1,
            use_cfc: false,
            seed: 7,
            ..Default::default()
        })
        .expect("snn");
        let inputs: Vec<Array1<f64>> = (0..5).map(|i| array![(i as f64) * 0.1, 0.0]).collect();
        let outs = snn.forward(&inputs).expect("fwd");
        assert_eq!(outs.len(), 5);
        for o in &outs {
            assert_eq!(o.len(), 1);
        }
    }

    #[test]
    fn test_liquid_snn_cfc() {
        let snn = LiquidSNN::<f64>::new(LiquidSNNConfig {
            input_dim: 1,
            reservoir_size: 4,
            output_dim: 2,
            use_cfc: true,
            seed: 8,
            ..Default::default()
        })
        .expect("snn cfc");
        let inputs: Vec<Array1<f64>> = (0..6).map(|_| array![0.5_f64]).collect();
        let outs = snn.forward(&inputs).expect("fwd");
        assert_eq!(outs.len(), 6);
        for o in &outs {
            assert_eq!(o.len(), 2);
        }
    }

    #[test]
    fn test_cfc_irregular() {
        let cell = CfCCell::<f64>::new(1, 4, 9);
        let times = vec![0.0_f64, 0.1, 0.5, 1.0, 2.3];
        let inputs: Vec<Array1<f64>> = times.iter().map(|_| array![1.0_f64]).collect();
        let outs = cell.forward_irregular(&times, &inputs).expect("irr");
        assert_eq!(outs.len(), 5);
    }
}
