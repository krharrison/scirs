//! ODE-RNN Hybrid: standard RNN with ODE between observations
//!
//! The ODE-RNN model (Rubanova et al., 2019) generalises classical RNNs to
//! **irregular** time grids by inserting an ODE solver between consecutive
//! observations.  The update rule is:
//!
//! ```text
//!   h̃_{i}  = ODESolve(f_θ, h_{i-1}, t_{i-1}, t_i)   (evolve between obs)
//!   h_i    = RNNCell(h̃_i, x_i)                        (update at obs)
//! ```
//!
//! ## GRU-ODE variant
//!
//! The GRU-ODE (de Brouwer et al., 2019) directly embeds GRU dynamics into
//! the continuous-time ODE, so the same function both propagates the hidden
//! state through time *and* acts as the discrete update:
//!
//! ```text
//!   dh/dt = GRUCell(h, 0) - h        (GRU update with zero input)
//! ```
//!
//! The ODE between observations is solved with fixed-step RK4.  At each
//! observation the state is updated with the true input via a full GRU step.

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid<F: Float + FromPrimitive>(x: F) -> F {
    let one = F::from(1.0).expect("1.0");
    one / (one + (-x).exp())
}

/// Dense linear layer `y = W x + b`.
fn linear<F: Float + FromPrimitive>(w: &Array2<F>, b: &Array1<F>, x: &Array1<F>) -> Array1<F> {
    let r = w.nrows();
    let mut y = Array1::zeros(r);
    for i in 0..r {
        let mut s = b[i];
        for j in 0..w.ncols() {
            s = s + w[[i, j]] * x[j];
        }
        y[i] = s;
    }
    y
}

/// RK4 integrator for `dh/dt = f(h)`.
fn rk4<F: Float + Debug + FromPrimitive + Clone>(
    f: &dyn Fn(&Array1<F>) -> Result<Array1<F>>,
    h0: &Array1<F>,
    n_steps: usize,
    dt: F,
) -> Result<Array1<F>> {
    if n_steps == 0 {
        return Ok(h0.clone());
    }
    let dim = h0.len();
    let half = F::from(0.5).expect("0.5");
    let sixth = F::from(1.0 / 6.0).expect("sixth");
    let two = F::from(2.0).expect("2");
    let mut h = h0.clone();

    for _ in 0..n_steps {
        let k1 = f(&h)?;
        let h2: Array1<F> = {
            let mut tmp = Array1::zeros(dim);
            for i in 0..dim {
                tmp[i] = h[i] + half * dt * k1[i];
            }
            tmp
        };
        let k2 = f(&h2)?;
        let h3: Array1<F> = {
            let mut tmp = Array1::zeros(dim);
            for i in 0..dim {
                tmp[i] = h[i] + half * dt * k2[i];
            }
            tmp
        };
        let k3 = f(&h3)?;
        let h4: Array1<F> = {
            let mut tmp = Array1::zeros(dim);
            for i in 0..dim {
                tmp[i] = h[i] + dt * k3[i];
            }
            tmp
        };
        let k4 = f(&h4)?;

        for i in 0..dim {
            h[i] = h[i] + dt * sixth * (k1[i] + two * k2[i] + two * k3[i] + k4[i]);
        }
    }
    Ok(h)
}

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

// ---------------------------------------------------------------------------
// GRU cell (shared between ODE-RNN and GRU-ODE)
// ---------------------------------------------------------------------------

/// Minimal GRU cell.
#[derive(Debug, Clone)]
pub struct GruCell<F: Float + Debug + FromPrimitive + Clone> {
    input_dim: usize,
    hidden_dim: usize,
    w_r: Array2<F>,
    w_u: Array2<F>,
    w_n: Array2<F>,
    b_r: Array1<F>,
    b_u: Array1<F>,
    b_n: Array1<F>,
}

impl<F: Float + Debug + FromPrimitive + Clone> GruCell<F> {
    /// Construct a GRU cell with Xavier/Glorot weight initialisation.
    pub fn new(input_dim: usize, hidden_dim: usize, seed: u64) -> Self {
        let concat = input_dim + hidden_dim;
        let std = F::from((2.0 / (concat + hidden_dim) as f64).sqrt()).expect("std");
        Self {
            input_dim,
            hidden_dim,
            w_r: random_matrix(hidden_dim, concat, std, seed),
            w_u: random_matrix(hidden_dim, concat, std, seed.wrapping_add(1)),
            w_n: random_matrix(hidden_dim, concat, std, seed.wrapping_add(2)),
            b_r: Array1::zeros(hidden_dim),
            b_u: Array1::zeros(hidden_dim),
            b_n: Array1::zeros(hidden_dim),
        }
    }

    /// Run a GRU step: `(x, h) → h'`.
    pub fn step(&self, x: &Array1<F>, h: &Array1<F>) -> Result<Array1<F>> {
        if x.len() != self.input_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.input_dim,
                actual: x.len(),
            });
        }
        if h.len() != self.hidden_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.hidden_dim,
                actual: h.len(),
            });
        }

        let concat = self.input_dim + self.hidden_dim;
        let mut xh = Array1::zeros(concat);
        for i in 0..self.input_dim {
            xh[i] = x[i];
        }
        for i in 0..self.hidden_dim {
            xh[self.input_dim + i] = h[i];
        }

        let r_raw = linear(&self.w_r, &self.b_r, &xh);
        let u_raw = linear(&self.w_u, &self.b_u, &xh);
        let mut r = Array1::zeros(self.hidden_dim);
        let mut u = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            r[i] = sigmoid(r_raw[i]);
            u[i] = sigmoid(u_raw[i]);
        }

        // candidate: [x; r ⊙ h]
        let mut xrh = Array1::zeros(concat);
        for i in 0..self.input_dim {
            xrh[i] = x[i];
        }
        for i in 0..self.hidden_dim {
            xrh[self.input_dim + i] = r[i] * h[i];
        }
        let n_raw = linear(&self.w_n, &self.b_n, &xrh);

        let one = F::one();
        let mut h_new = Array1::zeros(self.hidden_dim);
        for i in 0..self.hidden_dim {
            h_new[i] = (one - u[i]) * h[i] + u[i] * n_raw[i].tanh();
        }
        Ok(h_new)
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
// ODE dynamics for the hidden state
// ---------------------------------------------------------------------------

/// Small MLP acting as the ODE vector field for the hidden state.
///
/// `dh/dt = f_θ(h)` where `f_θ : ℝ^{hidden} → ℝ^{hidden}`.
#[derive(Debug, Clone)]
pub struct HiddenDynamics<F: Float + Debug + FromPrimitive + Clone> {
    hidden_dim: usize,
    w1: Array2<F>,
    b1: Array1<F>,
    w2: Array2<F>,
    b2: Array1<F>,
}

impl<F: Float + Debug + FromPrimitive + Clone> HiddenDynamics<F> {
    /// Construct a hidden ODE dynamics network.
    pub fn new(hidden_dim: usize, seed: u64) -> Self {
        let std = F::from((2.0 / (2 * hidden_dim) as f64).sqrt()).expect("std");
        Self {
            hidden_dim,
            w1: random_matrix(hidden_dim, hidden_dim, std, seed),
            b1: Array1::zeros(hidden_dim),
            w2: random_matrix(hidden_dim, hidden_dim, std, seed.wrapping_add(1)),
            b2: Array1::zeros(hidden_dim),
        }
    }

    /// Evaluate `f(h)` – the ODE vector field for the hidden state.
    pub fn forward(&self, h: &Array1<F>) -> Result<Array1<F>> {
        if h.len() != self.hidden_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.hidden_dim,
                actual: h.len(),
            });
        }
        let h1 = {
            let raw = linear(&self.w1, &self.b1, h);
            let mut out = Array1::zeros(self.hidden_dim);
            for (i, &v) in raw.iter().enumerate() {
                out[i] = v.tanh();
            }
            out
        };
        let h2 = {
            let raw = linear(&self.w2, &self.b2, &h1);
            let mut out = Array1::zeros(self.hidden_dim);
            for (i, &v) in raw.iter().enumerate() {
                out[i] = v.tanh();
            }
            out
        };
        Ok(h2)
    }
}

// ---------------------------------------------------------------------------
// ODE-RNN
// ---------------------------------------------------------------------------

/// Configuration for ODE-RNN.
#[derive(Debug, Clone)]
pub struct OdeRnnConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Hidden state size.
    pub hidden_dim: usize,
    /// Output dimensionality (readout layer).
    pub output_dim: usize,
    /// RK4 steps per unit time interval.
    pub ode_steps_per_unit: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for OdeRnnConfig {
    fn default() -> Self {
        Self {
            input_dim: 1,
            hidden_dim: 16,
            output_dim: 1,
            ode_steps_per_unit: 10,
            seed: 42,
        }
    }
}

/// ODE-RNN model for irregularly-sampled time series.
///
/// Between observations the hidden state evolves as an ODE;
/// at each observation it receives a standard GRU update.
///
/// ## Forward pass (single sequence)
///
/// ```text
/// h₀ = 0
/// for i in 1..N:
///     h̃_i  = RK4(f_θ, h_{i-1}, Δt_i)    // ODE propagation
///     h_i  = GRU(h̃_i, x_i)              // discrete update
/// ŷ = W_out h_N + b_out
/// ```
#[derive(Debug, Clone)]
pub struct OdeRnn<F: Float + Debug + FromPrimitive + Clone> {
    config: OdeRnnConfig,
    dynamics: HiddenDynamics<F>,
    gru: GruCell<F>,
    out_w: Array2<F>,
    out_b: Array1<F>,
}

impl<F: Float + Debug + FromPrimitive + Clone> OdeRnn<F> {
    /// Construct an ODE-RNN.
    pub fn new(config: OdeRnnConfig) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "input_dim".into(),
                message: "must be ≥ 1".into(),
            });
        }
        let s = config.seed;
        let std_out =
            F::from((2.0 / (config.hidden_dim + config.output_dim) as f64).sqrt()).expect("std");
        Ok(Self {
            dynamics: HiddenDynamics::new(config.hidden_dim, s),
            gru: GruCell::new(config.input_dim, config.hidden_dim, s.wrapping_add(10)),
            out_w: random_matrix(
                config.output_dim,
                config.hidden_dim,
                std_out,
                s.wrapping_add(20),
            ),
            out_b: Array1::zeros(config.output_dim),
            config,
        })
    }

    // ------------------------------------------------------------------

    fn ode_step(&self, h: &Array1<F>, dt: F) -> Result<Array1<F>> {
        let n_steps = ((dt.abs() * F::from(self.config.ode_steps_per_unit).expect("steps"))
            .ceil()
            .to_usize()
            .unwrap_or(1))
        .max(1);
        let step_dt = dt / F::from(n_steps).expect("n_steps");
        let dyn_ref = &self.dynamics;
        rk4(&|h_| dyn_ref.forward(h_), h, n_steps, step_dt)
    }

    fn readout(&self, h: &Array1<F>) -> Array1<F> {
        linear(&self.out_w, &self.out_b, h)
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Process an irregularly-sampled sequence.
    ///
    /// # Arguments
    /// * `times` – observation timestamps (monotonically non-decreasing).
    /// * `observations` – input vectors at those timestamps.
    ///
    /// # Returns
    /// Hidden state trajectory (one per observation).
    pub fn encode(&self, times: &[F], observations: &[Array1<F>]) -> Result<Vec<Array1<F>>> {
        if times.len() != observations.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: times.len(),
                actual: observations.len(),
            });
        }
        if observations.is_empty() {
            return Ok(Vec::new());
        }
        let n = observations.len();

        // Validate observation dimensions
        for obs in observations {
            if obs.len() != self.config.input_dim {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: self.config.input_dim,
                    actual: obs.len(),
                });
            }
        }

        let mut h = Array1::zeros(self.config.hidden_dim);
        // Initial GRU update (no ODE step before first observation)
        h = self.gru.step(&observations[0], &h)?;
        let mut hiddens = vec![h.clone()];

        for i in 1..n {
            // ODE propagation
            let dt = (times[i] - times[i - 1]).abs().max(F::epsilon());
            h = self.ode_step(&h, dt)?;
            // GRU update at observation
            h = self.gru.step(&observations[i], &h)?;
            hiddens.push(h.clone());
        }
        Ok(hiddens)
    }

    /// Predict output at each observation from the hidden state trajectory.
    pub fn predict_sequence(
        &self,
        times: &[F],
        observations: &[Array1<F>],
    ) -> Result<Vec<Array1<F>>> {
        let hiddens = self.encode(times, observations)?;
        Ok(hiddens.iter().map(|h| self.readout(h)).collect())
    }

    /// Return the final hidden state of the encoder.
    pub fn final_hidden(&self, times: &[F], observations: &[Array1<F>]) -> Result<Array1<F>> {
        let hiddens = self.encode(times, observations)?;
        hiddens
            .into_iter()
            .last()
            .ok_or_else(|| TimeSeriesError::InvalidInput("empty sequence".into()))
    }

    /// Predict one output from the final hidden state.
    pub fn predict(&self, times: &[F], observations: &[Array1<F>]) -> Result<Array1<F>> {
        let h = self.final_hidden(times, observations)?;
        Ok(self.readout(&h))
    }

    /// Return a reference to the config.
    pub fn config(&self) -> &OdeRnnConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// GRU-ODE
// ---------------------------------------------------------------------------

/// Configuration for GRU-ODE.
#[derive(Debug, Clone)]
pub struct GruOdeConfig {
    /// Input dimensionality.
    pub input_dim: usize,
    /// Hidden state size.
    pub hidden_dim: usize,
    /// Output dimensionality.
    pub output_dim: usize,
    /// RK4 steps per unit time when solving the continuous GRU ODE.
    pub ode_steps_per_unit: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for GruOdeConfig {
    fn default() -> Self {
        Self {
            input_dim: 1,
            hidden_dim: 16,
            output_dim: 1,
            ode_steps_per_unit: 10,
            seed: 42,
        }
    }
}

/// GRU-ODE model (de Brouwer et al., 2019).
///
/// The GRU dynamics are embedded directly in the ODE:
/// ```text
///   dh/dt = GRUCell(h, 0) - h
/// ```
/// so that the continuous trajectory follows the attractor defined by the
/// GRU update rule with a zero input.  Observations trigger a discrete GRU
/// update injecting the true input.
#[derive(Debug, Clone)]
pub struct GruOde<F: Float + Debug + FromPrimitive + Clone> {
    config: GruOdeConfig,
    /// GRU cell used both for continuous dynamics and discrete updates
    gru: GruCell<F>,
    out_w: Array2<F>,
    out_b: Array1<F>,
}

impl<F: Float + Debug + FromPrimitive + Clone> GruOde<F> {
    /// Construct a GRU-ODE model.
    pub fn new(config: GruOdeConfig) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "input_dim".into(),
                message: "must be ≥ 1".into(),
            });
        }
        let s = config.seed;
        let std_out =
            F::from((2.0 / (config.hidden_dim + config.output_dim) as f64).sqrt()).expect("std");
        Ok(Self {
            gru: GruCell::new(config.input_dim, config.hidden_dim, s),
            out_w: random_matrix(
                config.output_dim,
                config.hidden_dim,
                std_out,
                s.wrapping_add(30),
            ),
            out_b: Array1::zeros(config.output_dim),
            config,
        })
    }

    // ------------------------------------------------------------------

    /// Continuous GRU-ODE drift: `dh/dt = GRU(h, 0) - h`.
    fn continuous_drift(&self, h: &Array1<F>) -> Result<Array1<F>> {
        let zero_input = Array1::zeros(self.config.input_dim);
        let h_next = self.gru.step(&zero_input, h)?;
        let mut drift = Array1::zeros(self.config.hidden_dim);
        for i in 0..self.config.hidden_dim {
            drift[i] = h_next[i] - h[i];
        }
        Ok(drift)
    }

    /// Propagate hidden state from `t_prev` to `t_curr` via the continuous GRU-ODE.
    fn ode_propagate(&self, h: &Array1<F>, dt: F) -> Result<Array1<F>> {
        let n_steps = ((dt.abs() * F::from(self.config.ode_steps_per_unit).expect("steps"))
            .ceil()
            .to_usize()
            .unwrap_or(1))
        .max(1);
        let step_dt = dt / F::from(n_steps).expect("n_steps");
        let self_ref = self;
        rk4(&|h_| self_ref.continuous_drift(h_), h, n_steps, step_dt)
    }

    fn readout(&self, h: &Array1<F>) -> Array1<F> {
        linear(&self.out_w, &self.out_b, h)
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Encode an irregular time series, returning hidden state at each observation.
    ///
    /// Interleaves ODE propagation (between observations) with GRU updates
    /// (at each observation).
    pub fn encode(&self, times: &[F], observations: &[Array1<F>]) -> Result<Vec<Array1<F>>> {
        if times.len() != observations.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: times.len(),
                actual: observations.len(),
            });
        }
        if observations.is_empty() {
            return Ok(Vec::new());
        }

        for obs in observations {
            if obs.len() != self.config.input_dim {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: self.config.input_dim,
                    actual: obs.len(),
                });
            }
        }

        let n = observations.len();
        let mut h = Array1::zeros(self.config.hidden_dim);
        h = self.gru.step(&observations[0], &h)?;
        let mut hiddens = vec![h.clone()];

        for i in 1..n {
            // Continuous-time propagation between t[i-1] and t[i]
            let dt = (times[i] - times[i - 1]).abs().max(F::epsilon());
            h = self.ode_propagate(&h, dt)?;
            // Discrete GRU update at observation
            h = self.gru.step(&observations[i], &h)?;
            hiddens.push(h.clone());
        }
        Ok(hiddens)
    }

    /// Predict outputs at each observation from hidden state trajectory.
    pub fn predict_sequence(
        &self,
        times: &[F],
        observations: &[Array1<F>],
    ) -> Result<Vec<Array1<F>>> {
        let hiddens = self.encode(times, observations)?;
        Ok(hiddens.iter().map(|h| self.readout(h)).collect())
    }

    /// Predict a single output from the final hidden state.
    pub fn predict(&self, times: &[F], observations: &[Array1<F>]) -> Result<Array1<F>> {
        let hiddens = self.encode(times, observations)?;
        let h = hiddens
            .into_iter()
            .last()
            .ok_or_else(|| TimeSeriesError::InvalidInput("empty sequence".into()))?;
        Ok(self.readout(&h))
    }

    /// Return a reference to the config.
    pub fn config(&self) -> &GruOdeConfig {
        &self.config
    }

    /// Access the inner GRU cell.
    pub fn gru_cell(&self) -> &GruCell<F> {
        &self.gru
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_ode_rnn() -> OdeRnn<f64> {
        OdeRnn::new(OdeRnnConfig {
            input_dim: 2,
            hidden_dim: 8,
            output_dim: 1,
            ode_steps_per_unit: 5,
            seed: 0,
        })
        .expect("ode_rnn")
    }

    fn make_gru_ode() -> GruOde<f64> {
        GruOde::new(GruOdeConfig {
            input_dim: 2,
            hidden_dim: 8,
            output_dim: 1,
            ode_steps_per_unit: 5,
            seed: 1,
        })
        .expect("gru_ode")
    }

    #[test]
    fn test_ode_rnn_encode_shape() {
        let model = make_ode_rnn();
        let times = vec![0.0_f64, 0.5, 1.3, 2.1];
        let obs: Vec<Array1<f64>> = times.iter().map(|_| array![0.5_f64, -0.3]).collect();
        let hiddens = model.encode(&times, &obs).expect("encode");
        assert_eq!(hiddens.len(), 4);
        for h in &hiddens {
            assert_eq!(h.len(), 8);
        }
    }

    #[test]
    fn test_ode_rnn_predict() {
        let model = make_ode_rnn();
        let times = vec![0.0_f64, 1.0, 2.0];
        let obs: Vec<Array1<f64>> = times.iter().map(|_| array![1.0_f64, 0.0]).collect();
        let pred = model.predict(&times, &obs).expect("predict");
        assert_eq!(pred.len(), 1);
    }

    #[test]
    fn test_gru_ode_encode_shape() {
        let model = make_gru_ode();
        let times = vec![0.0_f64, 0.2, 1.0, 3.0];
        let obs: Vec<Array1<f64>> = times.iter().map(|_| array![0.1_f64, 0.9]).collect();
        let hiddens = model.encode(&times, &obs).expect("encode");
        assert_eq!(hiddens.len(), 4);
    }

    #[test]
    fn test_gru_ode_predict_sequence() {
        let model = make_gru_ode();
        let times = vec![0.0_f64, 0.1, 0.5, 1.0, 2.5];
        let obs: Vec<Array1<f64>> = times.iter().map(|i| array![*i, 1.0 - i]).collect();
        let preds = model.predict_sequence(&times, &obs).expect("predict_seq");
        assert_eq!(preds.len(), 5);
        for p in &preds {
            assert_eq!(p.len(), 1);
        }
    }

    #[test]
    fn test_gru_cell_step() {
        let cell = GruCell::<f64>::new(2, 4, 99);
        let x = array![0.5_f64, -0.3];
        let h = Array1::zeros(4);
        let h_new = cell.step(&x, &h).expect("step");
        assert_eq!(h_new.len(), 4);
    }

    #[test]
    fn test_dimension_error_ode_rnn() {
        let model = make_ode_rnn();
        let times = vec![0.0_f64, 1.0];
        // wrong input_dim: model expects 2, we send 3
        let obs = vec![array![1.0_f64, 2.0, 3.0], array![0.0_f64, 0.0, 0.0]];
        assert!(model.encode(&times, &obs).is_err());
    }

    #[test]
    fn test_irregular_large_gap() {
        // Large time gap between observations – ODE should still be stable
        let model = make_ode_rnn();
        let times = vec![0.0_f64, 0.001, 100.0];
        let obs: Vec<Array1<f64>> = times.iter().map(|_| array![0.0_f64, 0.0]).collect();
        let result = model.encode(&times, &obs);
        assert!(result.is_ok(), "Should handle large time gaps: {result:?}");
    }
}
