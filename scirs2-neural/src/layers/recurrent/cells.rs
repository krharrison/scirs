//! Single-step recurrent cell implementations
//!
//! This module provides single-step (cell-level) implementations of common
//! recurrent units. Cells process **one time-step** at a time and return the
//! updated hidden (and cell) state. For full-sequence processing use the
//! multi-step wrappers [`LSTM`] and [`GRU`].
//!
//! # Provided cells
//! - [`LSTMCell`] – Long Short-Term Memory cell (Hochreiter & Schmidhuber, 1997)
//! - [`GRUCell`]  – Gated Recurrent Unit cell (Cho et al., 2014)

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Distribution, Rng, Uniform};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Helper: sigmoid and tanh element-wise
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid<F: Float>(x: F) -> F {
    F::one() / (F::one() + (-x).exp())
}

// ---------------------------------------------------------------------------
// LSTMCell
// ---------------------------------------------------------------------------

/// Single-step Long Short-Term Memory cell.
///
/// Computes the following update equations at each time-step t:
///
/// ```text
/// i_t = σ(W_xi·x_t + b_xi + W_hi·h_{t-1} + b_hi)   (input gate)
/// f_t = σ(W_xf·x_t + b_xf + W_hf·h_{t-1} + b_hf)   (forget gate)
/// g_t = tanh(W_xg·x_t + b_xg + W_hg·h_{t-1} + b_hg)(cell input)
/// o_t = σ(W_xo·x_t + b_xo + W_ho·h_{t-1} + b_ho)   (output gate)
/// c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
/// h_t = o_t ⊙ tanh(c_t)
/// ```
///
/// # Examples
/// ```
/// use scirs2_neural::layers::recurrent::cells::LSTMCell;
/// use scirs2_core::ndarray::{Array, Array1};
/// use scirs2_core::random::rngs::SmallRng;
/// use scirs2_core::random::SeedableRng;
///
/// let mut rng = SmallRng::seed_from_u64(0);
/// let cell = LSTMCell::<f64>::new(10, 20, &mut rng).expect("build failed");
///
/// let x    = Array1::<f64>::zeros(10).into_dyn();
/// let h    = Array1::<f64>::zeros(20).into_dyn();
/// let c    = Array1::<f64>::zeros(20).into_dyn();
/// let (h_new, c_new) = cell.forward_step(&x, &h, &c).expect("step failed");
/// assert_eq!(h_new.shape(), &[20]);
/// assert_eq!(c_new.shape(), &[20]);
/// ```
pub struct LSTMCell<F: Float + Debug + Send + Sync + NumAssign> {
    input_size: usize,
    hidden_size: usize,
    // Input → gate weights (combined: [input_size, 4*hidden_size])
    weight_ih: Array2<F>,
    // Hidden → gate weights (combined: [hidden_size, 4*hidden_size])
    weight_hh: Array2<F>,
    // Input → gate biases ([4*hidden_size])
    bias_ih: Array1<F>,
    // Hidden → gate biases ([4*hidden_size])
    bias_hh: Array1<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> LSTMCell<F> {
    /// Create a new `LSTMCell` with Kaiming-uniform weight initialisation.
    pub fn new<R: Rng>(input_size: usize, hidden_size: usize, rng: &mut R) -> Result<Self> {
        if input_size == 0 || hidden_size == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "LSTMCell: sizes must be > 0".to_string(),
            ));
        }
        let scale_ih = (1.0_f64 / input_size as f64).sqrt();
        let scale_hh = (1.0_f64 / hidden_size as f64).sqrt();
        let four_h = 4 * hidden_size;

        let dist_ih = Uniform::new(-scale_ih, scale_ih)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("Uniform: {e}")))?;
        let dist_hh = Uniform::new(-scale_hh, scale_hh)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("Uniform: {e}")))?;

        let weight_ih = Array2::from_shape_fn((input_size, four_h), |_| {
            F::from(dist_ih.sample(rng)).unwrap_or(F::zero())
        });
        let weight_hh = Array2::from_shape_fn((hidden_size, four_h), |_| {
            F::from(dist_hh.sample(rng)).unwrap_or(F::zero())
        });
        let bias_ih = Array1::zeros(four_h);
        let bias_hh = Array1::zeros(four_h);

        Ok(Self {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        })
    }

    /// Single time-step forward pass.
    ///
    /// # Arguments
    /// * `x`   – Input vector, shape `[input_size]`
    /// * `h`   – Previous hidden state, shape `[hidden_size]`
    /// * `c`   – Previous cell state, shape `[hidden_size]`
    ///
    /// # Returns
    /// `(h_new, c_new)` – Updated hidden and cell states.
    pub fn forward_step(
        &self,
        x: &Array<F, IxDyn>,
        h: &Array<F, IxDyn>,
        c: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        let xshape = x.shape();
        let hshape = h.shape();
        let cshape = c.shape();
        if xshape != &[self.input_size] {
            return Err(NeuralError::InferenceError(format!(
                "LSTMCell: expected x shape [{:?}], got {:?}",
                self.input_size, xshape
            )));
        }
        if hshape != &[self.hidden_size] {
            return Err(NeuralError::InferenceError(format!(
                "LSTMCell: expected h shape [{:?}], got {:?}",
                self.hidden_size, hshape
            )));
        }
        if cshape != &[self.hidden_size] {
            return Err(NeuralError::InferenceError(format!(
                "LSTMCell: expected c shape [{:?}], got {:?}",
                self.hidden_size, cshape
            )));
        }
        let four_h = 4 * self.hidden_size;
        let mut gate = vec![F::zero(); four_h];
        // gate = W_ih^T · x + b_ih + W_hh^T · h + b_hh
        for j in 0..four_h {
            let mut val = self.bias_ih[j] + self.bias_hh[j];
            for i in 0..self.input_size {
                val = val + self.weight_ih[[i, j]] * x[i];
            }
            for i in 0..self.hidden_size {
                val = val + self.weight_hh[[i, j]] * h[i];
            }
            gate[j] = val;
        }
        let hs = self.hidden_size;
        // Split into i, f, g, o
        let i_gate: Vec<F> = gate[0..hs].iter().map(|&v| sigmoid(v)).collect();
        let f_gate: Vec<F> = gate[hs..2 * hs].iter().map(|&v| sigmoid(v)).collect();
        let g_gate: Vec<F> = gate[2 * hs..3 * hs].iter().map(|&v| v.tanh()).collect();
        let o_gate: Vec<F> = gate[3 * hs..4 * hs].iter().map(|&v| sigmoid(v)).collect();

        let mut c_new = vec![F::zero(); hs];
        let mut h_new = vec![F::zero(); hs];
        for k in 0..hs {
            c_new[k] = f_gate[k] * c[k] + i_gate[k] * g_gate[k];
            h_new[k] = o_gate[k] * c_new[k].tanh();
        }
        let c_out = Array::from_shape_vec(IxDyn(&[hs]), c_new)
            .map_err(|e| NeuralError::InferenceError(format!("shape err: {e}")))?;
        let h_out = Array::from_shape_vec(IxDyn(&[hs]), h_new)
            .map_err(|e| NeuralError::InferenceError(format!("shape err: {e}")))?;
        Ok((h_out, c_out))
    }

    /// Process a sequence of inputs `[seq_len, input_size]` starting from
    /// zero hidden and cell states.
    ///
    /// Returns all hidden states `[seq_len, hidden_size]`.
    pub fn forward_sequence(
        &self,
        xs: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let shape = xs.shape();
        if shape.len() != 2 || shape[1] != self.input_size {
            return Err(NeuralError::InferenceError(format!(
                "LSTMCell::forward_sequence: expected [seq_len, {}], got {:?}",
                self.input_size, shape
            )));
        }
        let seq_len = shape[0];
        let hs = self.hidden_size;
        let mut h = Array::<F, IxDyn>::zeros(IxDyn(&[hs]));
        let mut c = Array::<F, IxDyn>::zeros(IxDyn(&[hs]));
        let mut outputs = Vec::with_capacity(seq_len * hs);
        for t in 0..seq_len {
            let x_t = xs
                .slice(scirs2_core::ndarray::s![t, ..])
                .to_owned()
                .into_dyn();
            let (h_new, c_new) = self.forward_step(&x_t, &h, &c)?;
            outputs.extend_from_slice(
                h_new.as_slice().ok_or_else(|| {
                    NeuralError::InferenceError("Non-contiguous array".to_string())
                })?,
            );
            h = h_new;
            c = c_new;
        }
        Array::from_shape_vec(IxDyn(&[seq_len, hs]), outputs)
            .map_err(|e| NeuralError::InferenceError(format!("shape err: {e}")))
    }

    /// Input size of the cell.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Hidden size of the cell.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

// ---------------------------------------------------------------------------
// GRUCell
// ---------------------------------------------------------------------------

/// Single-step Gated Recurrent Unit cell.
///
/// Computes the following update equations at each time-step t:
///
/// ```text
/// r_t = σ(W_xr·x_t + b_xr + W_hr·h_{t-1} + b_hr)   (reset gate)
/// z_t = σ(W_xz·x_t + b_xz + W_hz·h_{t-1} + b_hz)   (update gate)
/// n_t = tanh(W_xn·x_t + b_xn + r_t ⊙ (W_hn·h_{t-1} + b_hn))  (new gate)
/// h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
/// ```
///
/// # Examples
/// ```
/// use scirs2_neural::layers::recurrent::cells::GRUCell;
/// use scirs2_core::ndarray::{Array, Array1};
/// use scirs2_core::random::rngs::SmallRng;
/// use scirs2_core::random::SeedableRng;
///
/// let mut rng = SmallRng::seed_from_u64(0);
/// let cell = GRUCell::<f64>::new(10, 20, &mut rng).expect("build failed");
///
/// let x   = Array1::<f64>::zeros(10).into_dyn();
/// let h   = Array1::<f64>::zeros(20).into_dyn();
/// let h_new = cell.forward_step(&x, &h).expect("step failed");
/// assert_eq!(h_new.shape(), &[20]);
/// ```
pub struct GRUCell<F: Float + Debug + Send + Sync + NumAssign> {
    input_size: usize,
    hidden_size: usize,
    // Combined input→gate weights [input_size, 3*hidden_size] (r, z, n)
    weight_ih: Array2<F>,
    // Combined hidden→gate weights [hidden_size, 3*hidden_size] (r, z, n)
    weight_hh: Array2<F>,
    bias_ih: Array1<F>,
    bias_hh: Array1<F>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> GRUCell<F> {
    /// Create a new `GRUCell`.
    pub fn new<R: Rng>(input_size: usize, hidden_size: usize, rng: &mut R) -> Result<Self> {
        if input_size == 0 || hidden_size == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "GRUCell: sizes must be > 0".to_string(),
            ));
        }
        let scale_ih = (1.0_f64 / input_size as f64).sqrt();
        let scale_hh = (1.0_f64 / hidden_size as f64).sqrt();
        let three_h = 3 * hidden_size;

        let dist_ih = Uniform::new(-scale_ih, scale_ih)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("Uniform: {e}")))?;
        let dist_hh = Uniform::new(-scale_hh, scale_hh)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("Uniform: {e}")))?;

        let weight_ih = Array2::from_shape_fn((input_size, three_h), |_| {
            F::from(dist_ih.sample(rng)).unwrap_or(F::zero())
        });
        let weight_hh = Array2::from_shape_fn((hidden_size, three_h), |_| {
            F::from(dist_hh.sample(rng)).unwrap_or(F::zero())
        });
        let bias_ih = Array1::zeros(three_h);
        let bias_hh = Array1::zeros(three_h);

        Ok(Self {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        })
    }

    /// Single time-step forward pass.
    ///
    /// # Arguments
    /// * `x` – Input vector, shape `[input_size]`
    /// * `h` – Previous hidden state, shape `[hidden_size]`
    ///
    /// # Returns
    /// `h_new` – Updated hidden state, shape `[hidden_size]`.
    pub fn forward_step(
        &self,
        x: &Array<F, IxDyn>,
        h: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let xshape = x.shape();
        let hshape = h.shape();
        if xshape != &[self.input_size] {
            return Err(NeuralError::InferenceError(format!(
                "GRUCell: expected x shape [{:?}], got {:?}",
                self.input_size, xshape
            )));
        }
        if hshape != &[self.hidden_size] {
            return Err(NeuralError::InferenceError(format!(
                "GRUCell: expected h shape [{:?}], got {:?}",
                self.hidden_size, hshape
            )));
        }
        let hs = self.hidden_size;
        let three_h = 3 * hs;

        // gates_x = W_ih^T · x + b_ih
        // gates_h = W_hh^T · h + b_hh
        let mut gates_x = vec![F::zero(); three_h];
        let mut gates_h = vec![F::zero(); three_h];
        for j in 0..three_h {
            let mut vx = self.bias_ih[j];
            let mut vh = self.bias_hh[j];
            for i in 0..self.input_size {
                vx = vx + self.weight_ih[[i, j]] * x[i];
            }
            for i in 0..hs {
                vh = vh + self.weight_hh[[i, j]] * h[i];
            }
            gates_x[j] = vx;
            gates_h[j] = vh;
        }

        // Reset and update gates use full (x + h) pre-activations
        let r_gate: Vec<F> = (0..hs)
            .map(|k| sigmoid(gates_x[k] + gates_h[k]))
            .collect();
        let z_gate: Vec<F> = (0..hs)
            .map(|k| sigmoid(gates_x[hs + k] + gates_h[hs + k]))
            .collect();

        // New gate: n = tanh(gates_x[2h..] + r ⊙ gates_h[2h..])
        let n_gate: Vec<F> = (0..hs)
            .map(|k| (gates_x[2 * hs + k] + r_gate[k] * gates_h[2 * hs + k]).tanh())
            .collect();

        // h_new = (1 - z) ⊙ n + z ⊙ h
        let h_new_vec: Vec<F> = (0..hs)
            .map(|k| (F::one() - z_gate[k]) * n_gate[k] + z_gate[k] * h[k])
            .collect();

        Array::from_shape_vec(IxDyn(&[hs]), h_new_vec)
            .map_err(|e| NeuralError::InferenceError(format!("shape err: {e}")))
    }

    /// Process a sequence of inputs `[seq_len, input_size]` from zero state.
    ///
    /// Returns all hidden states `[seq_len, hidden_size]`.
    pub fn forward_sequence(
        &self,
        xs: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        let shape = xs.shape();
        if shape.len() != 2 || shape[1] != self.input_size {
            return Err(NeuralError::InferenceError(format!(
                "GRUCell::forward_sequence: expected [seq_len, {}], got {:?}",
                self.input_size, shape
            )));
        }
        let seq_len = shape[0];
        let hs = self.hidden_size;
        let mut h = Array::<F, IxDyn>::zeros(IxDyn(&[hs]));
        let mut outputs = Vec::with_capacity(seq_len * hs);
        for t in 0..seq_len {
            let x_t = xs
                .slice(scirs2_core::ndarray::s![t, ..])
                .to_owned()
                .into_dyn();
            h = self.forward_step(&x_t, &h)?;
            outputs.extend_from_slice(
                h.as_slice().ok_or_else(|| {
                    NeuralError::InferenceError("Non-contiguous array".to_string())
                })?,
            );
        }
        Array::from_shape_vec(IxDyn(&[seq_len, hs]), outputs)
            .map_err(|e| NeuralError::InferenceError(format!("shape err: {e}")))
    }

    /// Input size of the cell.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Hidden size of the cell.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::SeedableRng;

    fn make_rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    // ---- LSTMCell ----

    #[test]
    fn test_lstm_cell_output_shape() {
        let mut rng = make_rng();
        let cell = LSTMCell::<f64>::new(5, 10, &mut rng).expect("build failed");
        let x = Array1::<f64>::zeros(5).into_dyn();
        let h = Array1::<f64>::zeros(10).into_dyn();
        let c = Array1::<f64>::zeros(10).into_dyn();
        let (h_new, c_new) = cell.forward_step(&x, &h, &c).expect("step failed");
        assert_eq!(h_new.shape(), &[10]);
        assert_eq!(c_new.shape(), &[10]);
    }

    #[test]
    fn test_lstm_cell_zero_input_bounded_output() {
        let mut rng = make_rng();
        let cell = LSTMCell::<f64>::new(4, 8, &mut rng).expect("build failed");
        let x = Array1::<f64>::zeros(4).into_dyn();
        let h = Array1::<f64>::zeros(8).into_dyn();
        let c = Array1::<f64>::zeros(8).into_dyn();
        let (h_new, _) = cell.forward_step(&x, &h, &c).expect("step failed");
        for &v in h_new.iter() {
            assert!(v.abs() <= 1.0, "LSTM h must be in [-1, 1], got {}", v);
        }
    }

    #[test]
    fn test_lstm_cell_wrong_input_size() {
        let mut rng = make_rng();
        let cell = LSTMCell::<f64>::new(5, 10, &mut rng).expect("build failed");
        let x = Array1::<f64>::zeros(6).into_dyn(); // wrong size
        let h = Array1::<f64>::zeros(10).into_dyn();
        let c = Array1::<f64>::zeros(10).into_dyn();
        assert!(cell.forward_step(&x, &h, &c).is_err());
    }

    #[test]
    fn test_lstm_cell_sequence() {
        let mut rng = make_rng();
        let cell = LSTMCell::<f64>::new(4, 8, &mut rng).expect("build failed");
        let xs = Array2::<f64>::from_elem((5, 4), 0.1).into_dyn();
        let outs = cell.forward_sequence(&xs).expect("sequence failed");
        assert_eq!(outs.shape(), &[5, 8]);
    }

    #[test]
    fn test_lstm_cell_accessors() {
        let mut rng = make_rng();
        let cell = LSTMCell::<f64>::new(6, 12, &mut rng).expect("build failed");
        assert_eq!(cell.input_size(), 6);
        assert_eq!(cell.hidden_size(), 12);
    }

    // ---- GRUCell ----

    #[test]
    fn test_gru_cell_output_shape() {
        let mut rng = make_rng();
        let cell = GRUCell::<f64>::new(5, 10, &mut rng).expect("build failed");
        let x = Array1::<f64>::zeros(5).into_dyn();
        let h = Array1::<f64>::zeros(10).into_dyn();
        let h_new = cell.forward_step(&x, &h).expect("step failed");
        assert_eq!(h_new.shape(), &[10]);
    }

    #[test]
    fn test_gru_cell_zero_state_bounded() {
        let mut rng = make_rng();
        let cell = GRUCell::<f64>::new(4, 8, &mut rng).expect("build failed");
        let x = Array1::<f64>::zeros(4).into_dyn();
        let h = Array1::<f64>::zeros(8).into_dyn();
        let h_new = cell.forward_step(&x, &h).expect("step failed");
        for &v in h_new.iter() {
            assert!(
                v.abs() <= 1.0,
                "GRU output should be in [-1, 1], got {}",
                v
            );
        }
    }

    #[test]
    fn test_gru_cell_wrong_input_size() {
        let mut rng = make_rng();
        let cell = GRUCell::<f64>::new(5, 10, &mut rng).expect("build failed");
        let x = Array1::<f64>::zeros(7).into_dyn(); // wrong
        let h = Array1::<f64>::zeros(10).into_dyn();
        assert!(cell.forward_step(&x, &h).is_err());
    }

    #[test]
    fn test_gru_cell_sequence() {
        let mut rng = make_rng();
        let cell = GRUCell::<f64>::new(4, 8, &mut rng).expect("build failed");
        let xs = Array2::<f64>::from_elem((6, 4), 0.2).into_dyn();
        let outs = cell.forward_sequence(&xs).expect("sequence failed");
        assert_eq!(outs.shape(), &[6, 8]);
    }

    #[test]
    fn test_gru_cell_accessors() {
        let mut rng = make_rng();
        let cell = GRUCell::<f64>::new(3, 7, &mut rng).expect("build failed");
        assert_eq!(cell.input_size(), 3);
        assert_eq!(cell.hidden_size(), 7);
    }

    // ---- Combined ----

    #[test]
    fn test_lstm_gru_different_hidden_sizes() {
        let mut rng = make_rng();
        let lstm = LSTMCell::<f64>::new(8, 16, &mut rng).expect("lstm build failed");
        let gru = GRUCell::<f64>::new(8, 16, &mut rng).expect("gru build failed");

        let x = Array1::<f64>::from_elem(8, 0.5).into_dyn();
        let h = Array1::<f64>::zeros(16).into_dyn();
        let c = Array1::<f64>::zeros(16).into_dyn();

        let (h_lstm, _) = lstm.forward_step(&x, &h, &c).expect("lstm step failed");
        let h_gru = gru.forward_step(&x, &h).expect("gru step failed");

        assert_eq!(h_lstm.shape(), &[16]);
        assert_eq!(h_gru.shape(), &[16]);
        // LSTM and GRU should produce different outputs
        let diff: f64 = h_lstm
            .iter()
            .zip(h_gru.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        assert!(diff > 0.0, "LSTM and GRU should produce different outputs");
    }

    #[test]
    fn test_lstm_cell_repeated_steps() {
        let mut rng = make_rng();
        let cell = LSTMCell::<f64>::new(3, 6, &mut rng).expect("build failed");
        let x = Array1::<f64>::from_elem(3, 0.1).into_dyn();
        let mut h = Array1::<f64>::zeros(6).into_dyn();
        let mut c = Array1::<f64>::zeros(6).into_dyn();
        // Run 10 steps, states should remain valid
        for _ in 0..10 {
            let (h_new, c_new) = cell.forward_step(&x, &h, &c).expect("step failed");
            for &v in h_new.iter() {
                assert!(v.is_finite(), "h must be finite");
                assert!(v.abs() <= 1.0, "h must be in [-1, 1]");
            }
            for &v in c_new.iter() {
                assert!(v.is_finite(), "c must be finite");
            }
            h = h_new;
            c = c_new;
        }
    }

    #[test]
    fn test_gru_cell_repeated_steps() {
        let mut rng = make_rng();
        let cell = GRUCell::<f64>::new(3, 6, &mut rng).expect("build failed");
        let x = Array1::<f64>::from_elem(3, 0.1).into_dyn();
        let mut h = Array1::<f64>::zeros(6).into_dyn();
        for _ in 0..10 {
            h = cell.forward_step(&x, &h).expect("step failed");
            for &v in h.iter() {
                assert!(v.is_finite(), "h must be finite");
            }
        }
    }
}
