//! Neural Controlled Differential Equations (Neural CDEs)
//!
//! A Neural CDE (Kidger et al., 2020) generalises Neural ODEs to time series by
//! driving the hidden state with a *controlled path* derived from the data:
//!
//! ```text
//!   dz(t) = f_θ(z(t)) dX(t)
//! ```
//!
//! where `X : [t₀, t_N] → ℝ^d` is a continuous interpolation of the
//! (possibly irregularly-sampled) observations, and `f_θ` is a learnable
//! *vector field network* whose output is a matrix of shape `[hidden × d]`.
//!
//! ## Implementation details
//!
//! * **Interpolation** – Cubic Hermite splines are fitted to the control path
//!   so that `X(t)` and `X'(t)` are available analytically.
//!
//! * **Log-signature features** – For rough-path theoretic analysis, the
//!   depth-2 log-signature of the control path is computed: it captures
//!   both the increment of the path and the *area enclosed* (Lévy area).
//!
//! * **ODE solver** – Fixed-step RK4 as in the Latent ODE module.
//!
//! * **Initial projection** – A linear map `z₀ = A x₀ + b` initialises the
//!   hidden state from the first observation.

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Cubic Hermite spline for the control path X(t)
// ---------------------------------------------------------------------------

/// Single-segment cubic Hermite spline between times `t0` and `t1`.
///
/// Coefficients `c0..c3` satisfy `p(τ) = c0 + c1 τ + c2 τ² + c3 τ³`
/// where `τ = (t - t0) / (t1 - t0)`.
#[derive(Debug, Clone)]
struct HermiteSegment<F: Float + Clone> {
    t0: F,
    t1: F,
    /// Coefficients for each channel: shape `[channels, 4]`
    coeffs: Array2<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive> HermiteSegment<F> {
    /// Evaluate `X(t)` for `t ∈ [t0, t1]`.
    fn eval(&self, t: F) -> Array1<F> {
        let h = self.t1 - self.t0;
        let tau = if h.abs() < F::epsilon() {
            F::zero()
        } else {
            (t - self.t0) / h
        };
        let tau2 = tau * tau;
        let tau3 = tau2 * tau;
        let channels = self.coeffs.nrows();
        let mut out = Array1::zeros(channels);
        for c in 0..channels {
            out[c] = self.coeffs[[c, 0]]
                + self.coeffs[[c, 1]] * tau
                + self.coeffs[[c, 2]] * tau2
                + self.coeffs[[c, 3]] * tau3;
        }
        out
    }

    /// Evaluate `dX/dt` (derivative) at time `t`.
    fn derivative(&self, t: F) -> Array1<F> {
        let h = self.t1 - self.t0;
        let tau = if h.abs() < F::epsilon() {
            F::zero()
        } else {
            (t - self.t0) / h
        };
        let tau2 = tau * tau;
        let inv_h = if h.abs() < F::epsilon() {
            F::zero()
        } else {
            F::one() / h
        };
        let channels = self.coeffs.nrows();
        let two = F::from(2.0).expect("2");
        let three = F::from(3.0).expect("3");
        let mut out = Array1::zeros(channels);
        for c in 0..channels {
            // d/dt [c0 + c1 τ + c2 τ² + c3 τ³]
            // = (c1 + 2 c2 τ + 3 c3 τ²) / h
            out[c] = (self.coeffs[[c, 1]]
                + two * self.coeffs[[c, 2]] * tau
                + three * self.coeffs[[c, 3]] * tau2)
                * inv_h;
        }
        out
    }
}

/// Cubic Hermite spline interpolation of a multi-channel control path.
///
/// Given `n` (time, observation) pairs, this struct provides `X(t)` and
/// `X'(t)` by fitting a Catmull-Rom–like tangent estimator at each knot.
#[derive(Debug, Clone)]
pub struct CubicHermiteSpline<F: Float + Debug + Clone + FromPrimitive> {
    segments: Vec<HermiteSegment<F>>,
    t_min: F,
    t_max: F,
    channels: usize,
}

impl<F: Float + Debug + Clone + FromPrimitive> CubicHermiteSpline<F> {
    /// Fit a cubic Hermite spline to the given (time, observation) sequence.
    ///
    /// Tangents are estimated using centered differences (Catmull-Rom).
    /// At the endpoints, one-sided differences are used.
    pub fn fit(times: &[F], values: &[Array1<F>]) -> Result<Self> {
        let n = times.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "CubicHermiteSpline requires at least 2 observations".into(),
                required: 2,
                actual: n,
            });
        }
        if values.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: values.len(),
            });
        }
        let channels = values[0].len();
        for v in values {
            if v.len() != channels {
                return Err(TimeSeriesError::InvalidInput(
                    "all observations must have the same dimensionality".into(),
                ));
            }
        }

        // Estimate tangents m_i
        let mut tangents: Vec<Array1<F>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut m = Array1::zeros(channels);
            if i == 0 {
                // Forward difference
                let dt = times[1] - times[0];
                if dt.abs() > F::epsilon() {
                    let inv_dt = F::one() / dt;
                    for c in 0..channels {
                        m[c] = (values[1][c] - values[0][c]) * inv_dt;
                    }
                }
            } else if i == n - 1 {
                // Backward difference
                let dt = times[n - 1] - times[n - 2];
                if dt.abs() > F::epsilon() {
                    let inv_dt = F::one() / dt;
                    for c in 0..channels {
                        m[c] = (values[n - 1][c] - values[n - 2][c]) * inv_dt;
                    }
                }
            } else {
                // Centered difference (Catmull-Rom)
                let dt = times[i + 1] - times[i - 1];
                if dt.abs() > F::epsilon() {
                    let inv_dt = F::one() / dt;
                    for c in 0..channels {
                        m[c] = (values[i + 1][c] - values[i - 1][c]) * inv_dt;
                    }
                }
            }
            tangents.push(m);
        }

        // Build segments
        let mut segments = Vec::with_capacity(n - 1);
        let two = F::from(2.0).expect("2");
        let three = F::from(3.0).expect("3");

        for i in 0..n - 1 {
            let h = times[i + 1] - times[i];
            // Hermite basis coefficients on [0, 1]:
            // p(τ) = (2τ³ - 3τ² + 1) p0 + (τ³ - 2τ² + τ) h m0
            //      + (-2τ³ + 3τ²) p1 + (τ³ - τ²) h m1
            // → c0 = p0
            //   c1 = h m0
            //   c2 = -3 p0 + 3 p1 - 2 h m0 - h m1
            //   c3 = 2 p0 - 2 p1 + h m0 + h m1
            let mut coeffs = Array2::zeros((channels, 4));
            for c in 0..channels {
                let p0 = values[i][c];
                let p1 = values[i + 1][c];
                let m0 = tangents[i][c] * h;
                let m1 = tangents[i + 1][c] * h;
                coeffs[[c, 0]] = p0;
                coeffs[[c, 1]] = m0;
                coeffs[[c, 2]] = -three * p0 + three * p1 - two * m0 - m1;
                coeffs[[c, 3]] = two * p0 - two * p1 + m0 + m1;
            }
            segments.push(HermiteSegment {
                t0: times[i],
                t1: times[i + 1],
                coeffs,
            });
        }

        Ok(Self {
            segments,
            t_min: times[0],
            t_max: times[n - 1],
            channels,
        })
    }

    /// Evaluate `X(t)`.  `t` is clamped to `[t_min, t_max]`.
    pub fn eval(&self, t: F) -> Array1<F> {
        let t_clamped = t.max(self.t_min).min(self.t_max);
        let seg = self.find_segment(t_clamped);
        self.segments[seg].eval(t_clamped)
    }

    /// Evaluate `dX/dt`.
    pub fn derivative(&self, t: F) -> Array1<F> {
        let t_clamped = t.max(self.t_min).min(self.t_max);
        let seg = self.find_segment(t_clamped);
        self.segments[seg].derivative(t_clamped)
    }

    fn find_segment(&self, t: F) -> usize {
        // Binary search for the active segment
        let n = self.segments.len();
        if n == 0 {
            return 0;
        }
        let mut lo = 0_usize;
        let mut hi = n - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if t <= self.segments[mid].t1 {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    /// Return the number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Return `(t_min, t_max)`.
    pub fn time_range(&self) -> (F, F) {
        (self.t_min, self.t_max)
    }
}

// ---------------------------------------------------------------------------
// Log-signature features (depth 2)
// ---------------------------------------------------------------------------

/// Depth-2 log-signature of a piecewise-linear path.
///
/// Given a path `X : [t₀, t_N] → ℝ^d` sampled at `n` points, the
/// depth-2 log-signature consists of:
///
/// * **Level-1**: the path increment  `ΔX = X(t_N) - X(t_0)` ∈ ℝ^d
/// * **Level-2**: the antisymmetric Lévy area matrix
///   `A_{ij} = ½ ∫(X_i - X_i(0)) dX_j - ∫(X_j - X_j(0)) dX_i`
///   approximated by the trapezoidal rule, flattened to the upper triangle.
///
/// The combined feature vector has dimension `d + d*(d-1)/2`.
#[derive(Debug, Clone)]
pub struct LogSignatureFeatures<F: Float + Clone> {
    /// Level-1: path increment
    pub level1: Array1<F>,
    /// Level-2: Lévy area (upper triangle, row-major)
    pub level2: Array1<F>,
    /// Number of input channels
    pub channels: usize,
}

impl<F: Float + Debug + Clone + FromPrimitive> LogSignatureFeatures<F> {
    /// Compute the depth-2 log-signature from observed (time, value) pairs.
    pub fn compute(times: &[F], values: &[Array1<F>]) -> Result<Self> {
        let n = times.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "log-signature requires at least 2 observations".into(),
                required: 2,
                actual: n,
            });
        }
        if values.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: values.len(),
            });
        }
        let d = values[0].len();

        // Level-1: global increment
        let mut level1 = Array1::zeros(d);
        for j in 0..d {
            level1[j] = values[n - 1][j] - values[0][j];
        }

        // Level-2: Lévy area via trapezoidal approximation
        // A_{ij} = ½ Σ_k [(X_i(k) - X_i(0))(ΔX_j(k)) - (X_j(k) - X_j(0))(ΔX_i(k))]
        let n_pairs = d * (d.saturating_sub(1)) / 2;
        let mut levy = Array2::<F>::zeros((d, d));
        let half = F::from(0.5).expect("0.5");

        for k in 0..n - 1 {
            let dt = times[k + 1] - times[k];
            if dt.abs() < F::epsilon() {
                continue;
            }
            for i in 0..d {
                for j in (i + 1)..d {
                    // Midpoint approximation of increments
                    let xi_mid = (values[k][i] + values[k + 1][i]) * half - values[0][i];
                    let xj_mid = (values[k][j] + values[k + 1][j]) * half - values[0][j];
                    let dxi = values[k + 1][i] - values[k][i];
                    let dxj = values[k + 1][j] - values[k][j];
                    let area = half * (xi_mid * dxj - xj_mid * dxi);
                    levy[[i, j]] = levy[[i, j]] + area;
                    levy[[j, i]] = levy[[j, i]] - area;
                }
            }
        }

        // Flatten upper triangle
        let mut level2 = Array1::zeros(n_pairs);
        let mut idx = 0;
        for i in 0..d {
            for j in (i + 1)..d {
                level2[idx] = levy[[i, j]];
                idx += 1;
            }
        }

        Ok(Self {
            level1,
            level2,
            channels: d,
        })
    }

    /// Return the concatenated `[level1 || level2]` feature vector.
    pub fn as_vector(&self) -> Array1<F> {
        let d = self.level1.len();
        let n_pairs = self.level2.len();
        let mut out = Array1::zeros(d + n_pairs);
        for i in 0..d {
            out[i] = self.level1[i];
        }
        for i in 0..n_pairs {
            out[d + i] = self.level2[i];
        }
        out
    }

    /// Total feature dimension: `d + d*(d-1)/2`.
    pub fn feature_dim(channels: usize) -> usize {
        channels + channels * channels.saturating_sub(1) / 2
    }
}

// ---------------------------------------------------------------------------
// CDE vector field network  f_θ(z) : ℝ^hidden → ℝ^{hidden × channels}
// ---------------------------------------------------------------------------

/// MLP parameterising the CDE vector field.
///
/// Output is reshaped to `[hidden_dim × control_channels]` so that
/// the update rule `dz = f(z) dX` can be applied as a matrix-vector product.
#[derive(Debug, Clone)]
pub struct CDEVectorField<F: Float + Debug + FromPrimitive + Clone> {
    hidden_dim: usize,
    control_channels: usize,
    /// Layer 1 weights [mlp_hidden × hidden_dim]
    w1: Array2<F>,
    b1: Array1<F>,
    /// Layer 2 weights [hidden_dim * control_channels × mlp_hidden]
    w2: Array2<F>,
    b2: Array1<F>,
}

impl<F: Float + Debug + FromPrimitive + Clone> CDEVectorField<F> {
    /// Create with MLP width `mlp_hidden`.
    pub fn new(
        hidden_dim: usize,
        control_channels: usize,
        mlp_hidden: usize,
        seed: u64,
    ) -> Self {
        let out_dim = hidden_dim * control_channels;
        let std1 = F::from((2.0 / (hidden_dim + mlp_hidden) as f64).sqrt()).expect("std");
        let std2 = F::from((2.0 / (mlp_hidden + out_dim) as f64).sqrt()).expect("std");
        Self {
            hidden_dim,
            control_channels,
            w1: random_matrix(mlp_hidden, hidden_dim, std1, seed),
            b1: Array1::zeros(mlp_hidden),
            w2: random_matrix(out_dim, mlp_hidden, std2, seed.wrapping_add(1)),
            b2: Array1::zeros(out_dim),
        }
    }

    /// Evaluate `f(z)` and return it as a `[hidden_dim × control_channels]` matrix.
    pub fn forward(&self, z: &Array1<F>) -> Result<Array2<F>> {
        if z.len() != self.hidden_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.hidden_dim,
                actual: z.len(),
            });
        }
        // Layer 1: tanh
        let h = {
            let mlp_h = self.w1.nrows();
            let mut h = Array1::zeros(mlp_h);
            for i in 0..mlp_h {
                let mut s = self.b1[i];
                for j in 0..self.hidden_dim {
                    s = s + self.w1[[i, j]] * z[j];
                }
                h[i] = s.tanh();
            }
            h
        };
        // Layer 2: tanh (bounded output keeps hidden state stable)
        let mlp_h = h.len();
        let out_dim = self.hidden_dim * self.control_channels;
        let mut flat = Array1::zeros(out_dim);
        for i in 0..out_dim {
            let mut s = self.b2[i];
            for j in 0..mlp_h {
                s = s + self.w2[[i, j]] * h[j];
            }
            flat[i] = s.tanh();
        }
        // Reshape to [hidden_dim × control_channels]
        let mut mat = Array2::zeros((self.hidden_dim, self.control_channels));
        for i in 0..self.hidden_dim {
            for j in 0..self.control_channels {
                mat[[i, j]] = flat[i * self.control_channels + j];
            }
        }
        Ok(mat)
    }
}

// ---------------------------------------------------------------------------
// RK4 integrator for the CDE
// ---------------------------------------------------------------------------

/// Integrate `dz = f(z) dX` using a fixed-step RK4 scheme.
///
/// At each sub-step `t_k`, the vector field matrix `F_k = f(z(t_k))` is
/// evaluated and multiplied by the path derivative `X'(t_k)` to obtain the
/// effective drift `dz/dt = F(z) X'(t)`.
fn cde_rk4<F: Float + Debug + FromPrimitive + Clone>(
    field: &CDEVectorField<F>,
    spline: &CubicHermiteSpline<F>,
    z0: &Array1<F>,
    t0: F,
    t1: F,
    n_steps: usize,
) -> Result<Array1<F>> {
    if n_steps == 0 {
        return Ok(z0.clone());
    }
    let dim = z0.len();
    let dt = (t1 - t0) / F::from(n_steps).expect("n_steps");
    let half = F::from(0.5).expect("0.5");
    let sixth = F::from(1.0 / 6.0).expect("sixth");
    let two = F::from(2.0).expect("2");

    // Compute the CDE drift at a given (z, t): f(z) dX/dt
    let effective_drift =
        |z: &Array1<F>, t: F| -> Result<Array1<F>> {
            let fmat = field.forward(z)?;
            let dxt = spline.derivative(t);
            let mut drift = Array1::zeros(dim);
            for i in 0..dim {
                let mut s = F::zero();
                for j in 0..fmat.ncols() {
                    s = s + fmat[[i, j]] * dxt[j];
                }
                drift[i] = s;
            }
            Ok(drift)
        };

    let mut z = z0.clone();
    for step in 0..n_steps {
        let t_k = t0 + F::from(step).expect("step") * dt;
        let t_k_half = t_k + half * dt;
        let t_k_next = t_k + dt;

        let k1 = effective_drift(&z, t_k)?;

        let z2: Array1<F> = {
            let mut tmp = Array1::zeros(dim);
            for i in 0..dim {
                tmp[i] = z[i] + half * dt * k1[i];
            }
            tmp
        };
        let k2 = effective_drift(&z2, t_k_half)?;

        let z3: Array1<F> = {
            let mut tmp = Array1::zeros(dim);
            for i in 0..dim {
                tmp[i] = z[i] + half * dt * k2[i];
            }
            tmp
        };
        let k3 = effective_drift(&z3, t_k_half)?;

        let z4: Array1<F> = {
            let mut tmp = Array1::zeros(dim);
            for i in 0..dim {
                tmp[i] = z[i] + dt * k3[i];
            }
            tmp
        };
        let k4 = effective_drift(&z4, t_k_next)?;

        for i in 0..dim {
            z[i] = z[i] + dt * sixth * (k1[i] + two * k2[i] + two * k3[i] + k4[i]);
        }
    }
    Ok(z)
}

// ---------------------------------------------------------------------------
// Neural CDE model
// ---------------------------------------------------------------------------

/// Configuration for the Neural CDE model.
#[derive(Debug, Clone)]
pub struct NeuralCDEConfig {
    /// Number of input channels (observation dimension).
    pub input_channels: usize,
    /// Hidden state dimension.
    pub hidden_dim: usize,
    /// Output dimension (prediction target).
    pub output_dim: usize,
    /// Width of the MLP inside the CDE vector field.
    pub mlp_hidden: usize,
    /// Number of RK4 steps per unit time for the CDE integration.
    pub ode_steps_per_unit: usize,
    /// Use log-signature features for rough-path analysis.
    pub use_log_signature: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for NeuralCDEConfig {
    fn default() -> Self {
        Self {
            input_channels: 1,
            hidden_dim: 16,
            output_dim: 1,
            mlp_hidden: 32,
            ode_steps_per_unit: 10,
            use_log_signature: false,
            seed: 42,
        }
    }
}

/// Neural Controlled Differential Equation model.
///
/// The model integrates `dz(t) = f_θ(z(t)) dX(t)` where `X(t)` is the
/// cubic Hermite spline through the observations.  The final hidden state
/// `z(t_N)` is projected linearly to the output space.
///
/// Additionally, depth-2 log-signature features can be extracted from the
/// control path for downstream rough-path analysis.
#[derive(Debug, Clone)]
pub struct NeuralCDE<F: Float + Debug + FromPrimitive + Clone> {
    config: NeuralCDEConfig,
    /// Initial projection: z₀ = W x₀ + b
    init_w: Array2<F>,
    init_b: Array1<F>,
    /// CDE vector field
    vector_field: CDEVectorField<F>,
    /// Output projection: ŷ = W_out z(t_N) + b_out
    out_w: Array2<F>,
    out_b: Array1<F>,
}

impl<F: Float + Debug + FromPrimitive + Clone> NeuralCDE<F> {
    /// Construct a Neural CDE with the given configuration.
    pub fn new(config: NeuralCDEConfig) -> Result<Self> {
        if config.input_channels == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "input_channels".into(),
                message: "must be ≥ 1".into(),
            });
        }
        if config.hidden_dim == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "hidden_dim".into(),
                message: "must be ≥ 1".into(),
            });
        }

        let s = config.seed;
        let std_init = F::from(
            (2.0 / (config.input_channels + config.hidden_dim) as f64).sqrt(),
        )
        .expect("std");
        let std_out = F::from(
            (2.0 / (config.hidden_dim + config.output_dim) as f64).sqrt(),
        )
        .expect("std");

        Ok(Self {
            vector_field: CDEVectorField::new(
                config.hidden_dim,
                config.input_channels,
                config.mlp_hidden,
                s,
            ),
            init_w: random_matrix(config.hidden_dim, config.input_channels, std_init, s.wrapping_add(50)),
            init_b: Array1::zeros(config.hidden_dim),
            out_w: random_matrix(config.output_dim, config.hidden_dim, std_out, s.wrapping_add(60)),
            out_b: Array1::zeros(config.output_dim),
            config,
        })
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn project_init(&self, x0: &Array1<F>) -> Result<Array1<F>> {
        if x0.len() != self.config.input_channels {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.input_channels,
                actual: x0.len(),
            });
        }
        let h = self.config.hidden_dim;
        let mut z0 = Array1::zeros(h);
        for i in 0..h {
            let mut s = self.init_b[i];
            for j in 0..self.config.input_channels {
                s = s + self.init_w[[i, j]] * x0[j];
            }
            z0[i] = s.tanh();
        }
        Ok(z0)
    }

    fn project_output(&self, z: &Array1<F>) -> Array1<F> {
        let d = self.config.output_dim;
        let mut out = Array1::zeros(d);
        for i in 0..d {
            let mut s = self.out_b[i];
            for j in 0..self.config.hidden_dim {
                s = s + self.out_w[[i, j]] * z[j];
            }
            out[i] = s;
        }
        out
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Solve the CDE and return the **final hidden state** `z(t_N)`.
    ///
    /// # Arguments
    /// * `times` – observation timestamps (monotonically non-decreasing, ≥ 2 points).
    /// * `observations` – one `Array1` per timestamp, all of dimension `input_channels`.
    pub fn solve(
        &self,
        times: &[F],
        observations: &[Array1<F>],
    ) -> Result<Array1<F>> {
        if times.len() != observations.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: times.len(),
                actual: observations.len(),
            });
        }
        let spline = CubicHermiteSpline::fit(times, observations)?;
        let z0 = self.project_init(&observations[0])?;

        let t0 = times[0];
        let t1 = times[times.len() - 1];
        let duration = (t1 - t0).abs();
        let n_steps = ((duration
            * F::from(self.config.ode_steps_per_unit).expect("steps"))
        .ceil()
        .to_usize()
        .unwrap_or(1))
        .max(1);

        cde_rk4(
            &self.vector_field,
            &spline,
            &z0,
            t0,
            t1,
            n_steps,
        )
    }

    /// Forward pass: solve the CDE, then project `z(t_N)` to output space.
    ///
    /// Returns a prediction vector of dimension `output_dim`.
    pub fn forward(
        &self,
        times: &[F],
        observations: &[Array1<F>],
    ) -> Result<Array1<F>> {
        let z_final = self.solve(times, observations)?;
        Ok(self.project_output(&z_final))
    }

    /// Compute depth-2 log-signature features from the control path.
    ///
    /// Returns `None` if `use_log_signature` is `false` in the config.
    pub fn log_signature_features(
        &self,
        times: &[F],
        observations: &[Array1<F>],
    ) -> Result<Option<LogSignatureFeatures<F>>> {
        if !self.config.use_log_signature {
            return Ok(None);
        }
        LogSignatureFeatures::compute(times, observations).map(Some)
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &NeuralCDEConfig {
        &self.config
    }

    /// Access the internal spline for a given set of observations.
    pub fn build_spline(
        &self,
        times: &[F],
        observations: &[Array1<F>],
    ) -> Result<CubicHermiteSpline<F>> {
        CubicHermiteSpline::fit(times, observations)
    }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_model() -> NeuralCDE<f64> {
        NeuralCDE::new(NeuralCDEConfig {
            input_channels: 2,
            hidden_dim: 8,
            output_dim: 1,
            mlp_hidden: 16,
            ode_steps_per_unit: 5,
            use_log_signature: true,
            seed: 1,
        })
        .expect("model")
    }

    #[test]
    fn test_spline_eval() {
        let times = vec![0.0_f64, 1.0, 2.0, 3.0];
        let values: Vec<Array1<f64>> = vec![
            array![0.0, 0.0],
            array![1.0, 0.5],
            array![0.0, 1.0],
            array![-1.0, 0.0],
        ];
        let spline = CubicHermiteSpline::fit(&times, &values).expect("spline");
        // At knot points, eval should match the given values closely
        let v = spline.eval(0.0);
        assert!((v[0] - 0.0).abs() < 1e-10);
        let v = spline.eval(1.0);
        assert!((v[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_signature_shape() {
        let times = vec![0.0_f64, 0.5, 1.0, 1.5];
        let obs: Vec<Array1<f64>> = vec![
            array![0.0, 0.0],
            array![1.0, 0.5],
            array![2.0, 1.0],
            array![3.0, 0.5],
        ];
        let lsf = LogSignatureFeatures::compute(&times, &obs).expect("lsf");
        // d=2: level1=2, level2=1 → total=3
        assert_eq!(lsf.level1.len(), 2);
        assert_eq!(lsf.level2.len(), 1);
        let fv = lsf.as_vector();
        assert_eq!(fv.len(), LogSignatureFeatures::<f64>::feature_dim(2));
    }

    #[test]
    fn test_forward_shape() {
        let model = make_model();
        let times = vec![0.0_f64, 1.0, 2.0];
        let obs: Vec<Array1<f64>> = vec![
            array![0.0, 0.0],
            array![1.0, 0.5],
            array![0.0, 1.0],
        ];
        let pred = model.forward(&times, &obs).expect("forward");
        assert_eq!(pred.len(), 1);
    }

    #[test]
    fn test_log_signature_feature() {
        let model = make_model();
        let times = vec![0.0_f64, 1.0];
        let obs: Vec<Array1<f64>> = vec![array![0.0, 0.0], array![1.0, 1.0]];
        let lsf = model
            .log_signature_features(&times, &obs)
            .expect("lsf")
            .expect("some");
        assert_eq!(lsf.channels, 2);
    }

    #[test]
    fn test_insufficient_data_error() {
        let times = vec![0.0_f64];
        let obs = vec![array![1.0_f64, 2.0]];
        let result = CubicHermiteSpline::fit(&times, &obs);
        assert!(result.is_err());
    }
}
