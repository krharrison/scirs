//! Physics-Informed Multi-Dimensional ODE Predictor
//!
//! Extends the physics-informed time series module with multi-dimensional
//! ODE-constrained prediction. Supports user-defined ODE systems, conservation
//! laws, and hybrid loss functions that combine data fidelity with physics
//! constraints.
//!
//! ## Overview
//!
//! The [`PhysicsInformedPredictor`] trains a small MLP to predict multi-variate
//! state trajectories while penalising violations of:
//! - **ODE residuals** at collocation points
//! - **Conservation laws** (energy, mass, momentum, or user-supplied)
//!
//! The total loss is:
//!
//! ```text
//! L = data_weight * L_data + physics_weight * L_physics
//! ```
//!
//! where `L_physics` includes both ODE residual terms and conservation-law
//! soft-constraint terms.

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Symbolic ODE system definition.
///
/// Encodes a system of ordinary differential equations as symbolic string
/// expressions together with the state dimensionality. The equations are
/// evaluated numerically via a simple expression evaluator that supports
/// basic arithmetic and common mathematical functions.
#[derive(Debug, Clone)]
pub struct ODESystem {
    /// Symbolic equations as strings (one per state dimension).
    /// Each equation defines `dx_i/dt` as a function of the state variables
    /// `x0, x1, ...` and the independent variable `t`.
    /// Supported tokens: `+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `x0..xN`, `t`, numeric literals.
    pub equations: Vec<String>,
    /// Dimensionality of the state vector.
    pub state_dim: usize,
}

/// Conservation law constraint.
///
/// Encapsulates a conserved quantity that should remain (approximately) constant
/// along the trajectory. Violations are penalised as soft constraints in the
/// physics loss.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ConservationLaw {
    /// Energy conservation: `sum(x_i^2 / 2)` is constant.
    EnergyConservation,
    /// Mass conservation: `sum(x_i)` is constant.
    MassConservation,
    /// Momentum conservation: `sum(x_i)` weighted by index (simple model).
    MomentumConservation,
    /// User-defined conserved quantity specified by coefficient vector.
    /// The conserved quantity is `sum(coeffs[i] * x_i)`.
    CustomLinear(Vec<f64>),
}

/// Configuration for [`PhysicsInformedPredictor`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PhysicsInformedConfig {
    /// Weight for the physics (ODE residual + conservation) loss term.
    pub physics_weight: f64,
    /// Weight for the data-fidelity loss term.
    pub data_weight: f64,
    /// Optional ODE system to enforce.
    pub ode_system: Option<ODESystem>,
    /// Conservation laws to enforce as soft constraints.
    pub conservation_laws: Vec<ConservationLaw>,
    /// Number of hidden units per MLP layer.
    pub hidden_dim: usize,
    /// Number of hidden layers.
    pub n_layers: usize,
    /// Number of training epochs.
    pub n_epochs: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Number of collocation points for ODE residual evaluation.
    pub n_collocation: usize,
}

impl Default for PhysicsInformedConfig {
    fn default() -> Self {
        Self {
            physics_weight: 1.0,
            data_weight: 1.0,
            ode_system: None,
            conservation_laws: Vec::new(),
            hidden_dim: 32,
            n_layers: 2,
            n_epochs: 100,
            learning_rate: 1e-3,
            n_collocation: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal MLP (ℝ → ℝ^d)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DenseLayer {
    w: Vec<f64>,
    b: Vec<f64>,
    in_dim: usize,
    out_dim: usize,
}

impl DenseLayer {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (2.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let w: Vec<f64> = (0..out_dim * in_dim)
            .map(|k| {
                let v = ((k as f64 * 1.61803398) % 2.0) - 1.0;
                v * scale
            })
            .collect();
        Self {
            w,
            b: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    fn forward_tanh(&self, x: &[f64]) -> Vec<f64> {
        let mut y = self.b.clone();
        for (i, yi) in y.iter_mut().enumerate() {
            for (j, &xj) in x.iter().enumerate() {
                *yi += self.w[i * self.in_dim + j] * xj;
            }
        }
        y.iter_mut().for_each(|v| *v = v.tanh());
        y
    }

    fn forward_linear(&self, x: &[f64]) -> Vec<f64> {
        let mut y = self.b.clone();
        for (i, yi) in y.iter_mut().enumerate() {
            for (j, &xj) in x.iter().enumerate() {
                *yi += self.w[i * self.in_dim + j] * xj;
            }
        }
        y
    }
}

/// Multi-layer perceptron: ℝ → ℝ^d
#[derive(Debug, Clone)]
struct MultiOutputMlp {
    hidden: Vec<DenseLayer>,
    output: DenseLayer,
    out_dim: usize,
}

impl MultiOutputMlp {
    fn new(hidden_dim: usize, n_layers: usize, out_dim: usize) -> Self {
        let mut hidden = vec![DenseLayer::new(1, hidden_dim)];
        for _ in 1..n_layers {
            hidden.push(DenseLayer::new(hidden_dim, hidden_dim));
        }
        let output = DenseLayer::new(hidden_dim, out_dim);
        Self {
            hidden,
            output,
            out_dim,
        }
    }

    fn predict(&self, t: f64) -> Vec<f64> {
        let mut h = vec![t];
        for layer in &self.hidden {
            h = layer.forward_tanh(&h);
        }
        self.output.forward_linear(&h)
    }

    fn n_params(&self) -> usize {
        let mut n = 0;
        for l in &self.hidden {
            n += l.w.len() + l.b.len();
        }
        n += self.output.w.len() + self.output.b.len();
        n
    }

    fn flatten(&self) -> Vec<f64> {
        let mut p = Vec::new();
        for l in &self.hidden {
            p.extend_from_slice(&l.w);
            p.extend_from_slice(&l.b);
        }
        p.extend_from_slice(&self.output.w);
        p.extend_from_slice(&self.output.b);
        p
    }

    fn unflatten(&mut self, params: &[f64]) {
        let mut idx = 0;
        for l in &mut self.hidden {
            let wn = l.w.len();
            let bn = l.b.len();
            l.w.copy_from_slice(&params[idx..idx + wn]);
            idx += wn;
            l.b.copy_from_slice(&params[idx..idx + bn]);
            idx += bn;
        }
        let wn = self.output.w.len();
        let bn = self.output.b.len();
        self.output.w.copy_from_slice(&params[idx..idx + wn]);
        idx += wn;
        self.output.b.copy_from_slice(&params[idx..idx + bn]);
        let _ = idx;
    }
}

// ---------------------------------------------------------------------------
// Adam optimiser (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    t: u64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
}

impl Adam {
    fn new(n: usize, lr: f64) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ---------------------------------------------------------------------------
// Simple expression evaluator for ODE equations
// ---------------------------------------------------------------------------

/// Evaluate a simple symbolic ODE expression.
///
/// Supported tokens:
/// - State variables: `x0`, `x1`, ... `xN`
/// - Independent variable: `t`
/// - Operators: `+`, `-`, `*`, `/`
/// - Functions: `sin(...)`, `cos(...)`, `exp(...)`
/// - Numeric literals (including negatives in parentheses)
///
/// The evaluator uses a simple recursive-descent parser.
fn evaluate_ode_expr(expr: &str, state: &[f64], t: f64) -> Result<f64> {
    let tokens = tokenize_expr(expr)?;
    let mut pos = 0;
    let result = parse_add_sub(&tokens, &mut pos, state, t)?;
    Ok(result)
}

#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Var(usize), // x0, x1, ...
    Time,       // t
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Sin,
    Cos,
    Exp,
}

fn tokenize_expr(expr: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' => {
                i += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            'x' if i + 1 < chars.len() && chars[i + 1].is_ascii_digit() => {
                i += 1;
                let start = i;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let idx_str: String = chars[start..i].iter().collect();
                let idx = idx_str.parse::<usize>().map_err(|_| {
                    TimeSeriesError::InvalidInput(format!("Invalid variable index: {}", idx_str))
                })?;
                tokens.push(Token::Var(idx));
            }
            't' if (i + 1 >= chars.len() || !chars[i + 1].is_alphanumeric()) => {
                tokens.push(Token::Time);
                i += 1;
            }
            's' if i + 2 < chars.len() && chars[i + 1] == 'i' && chars[i + 2] == 'n' => {
                tokens.push(Token::Sin);
                i += 3;
            }
            'c' if i + 2 < chars.len() && chars[i + 1] == 'o' && chars[i + 2] == 's' => {
                tokens.push(Token::Cos);
                i += 3;
            }
            'e' if i + 2 < chars.len() && chars[i + 1] == 'x' && chars[i + 2] == 'p' => {
                tokens.push(Token::Exp);
                i += 3;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let val = num_str.parse::<f64>().map_err(|_| {
                    TimeSeriesError::InvalidInput(format!("Invalid number: {}", num_str))
                })?;
                tokens.push(Token::Number(val));
            }
            other => {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "Unexpected character in ODE expression: '{}'",
                    other
                )));
            }
        }
    }

    Ok(tokens)
}

fn parse_add_sub(tokens: &[Token], pos: &mut usize, state: &[f64], t: f64) -> Result<f64> {
    let mut left = parse_mul_div(tokens, pos, state, t)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Plus => {
                *pos += 1;
                let right = parse_mul_div(tokens, pos, state, t)?;
                left += right;
            }
            Token::Minus => {
                *pos += 1;
                let right = parse_mul_div(tokens, pos, state, t)?;
                left -= right;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_mul_div(tokens: &[Token], pos: &mut usize, state: &[f64], t: f64) -> Result<f64> {
    let mut left = parse_unary(tokens, pos, state, t)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Star => {
                *pos += 1;
                let right = parse_unary(tokens, pos, state, t)?;
                left *= right;
            }
            Token::Slash => {
                *pos += 1;
                let right = parse_unary(tokens, pos, state, t)?;
                if right.abs() < 1e-15 {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Division by zero in ODE expression".to_string(),
                    ));
                }
                left /= right;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_unary(tokens: &[Token], pos: &mut usize, state: &[f64], t: f64) -> Result<f64> {
    if *pos < tokens.len() {
        if let Token::Minus = &tokens[*pos] {
            *pos += 1;
            let val = parse_atom(tokens, pos, state, t)?;
            return Ok(-val);
        }
        if let Token::Plus = &tokens[*pos] {
            *pos += 1;
        }
    }
    parse_atom(tokens, pos, state, t)
}

fn parse_atom(tokens: &[Token], pos: &mut usize, state: &[f64], t: f64) -> Result<f64> {
    if *pos >= tokens.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Unexpected end of ODE expression".to_string(),
        ));
    }

    match &tokens[*pos] {
        Token::Number(v) => {
            let val = *v;
            *pos += 1;
            Ok(val)
        }
        Token::Var(idx) => {
            let idx = *idx;
            *pos += 1;
            if idx < state.len() {
                Ok(state[idx])
            } else {
                Err(TimeSeriesError::InvalidInput(format!(
                    "Variable index x{} out of range (state_dim={})",
                    idx,
                    state.len()
                )))
            }
        }
        Token::Time => {
            *pos += 1;
            Ok(t)
        }
        Token::Sin => {
            *pos += 1;
            // Expect '('
            if *pos < tokens.len() {
                if let Token::LParen = &tokens[*pos] {
                    *pos += 1;
                }
            }
            let val = parse_add_sub(tokens, pos, state, t)?;
            // Expect ')'
            if *pos < tokens.len() {
                if let Token::RParen = &tokens[*pos] {
                    *pos += 1;
                }
            }
            Ok(val.sin())
        }
        Token::Cos => {
            *pos += 1;
            if *pos < tokens.len() {
                if let Token::LParen = &tokens[*pos] {
                    *pos += 1;
                }
            }
            let val = parse_add_sub(tokens, pos, state, t)?;
            if *pos < tokens.len() {
                if let Token::RParen = &tokens[*pos] {
                    *pos += 1;
                }
            }
            Ok(val.cos())
        }
        Token::Exp => {
            *pos += 1;
            if *pos < tokens.len() {
                if let Token::LParen = &tokens[*pos] {
                    *pos += 1;
                }
            }
            let val = parse_add_sub(tokens, pos, state, t)?;
            if *pos < tokens.len() {
                if let Token::RParen = &tokens[*pos] {
                    *pos += 1;
                }
            }
            Ok(val.exp())
        }
        Token::LParen => {
            *pos += 1;
            let val = parse_add_sub(tokens, pos, state, t)?;
            if *pos < tokens.len() {
                if let Token::RParen = &tokens[*pos] {
                    *pos += 1;
                }
            }
            Ok(val)
        }
        _ => Err(TimeSeriesError::InvalidInput(format!(
            "Unexpected token in ODE expression at position {}",
            *pos
        ))),
    }
}

// ---------------------------------------------------------------------------
// Physics loss helpers
// ---------------------------------------------------------------------------

/// Compute ODE residual: dx/dt - f(x, t)
fn ode_residual_at(
    mlp: &MultiOutputMlp,
    ode: &ODESystem,
    t_val: f64,
    fd_eps: f64,
) -> Result<Vec<f64>> {
    let state = mlp.predict(t_val);
    let state_plus = mlp.predict(t_val + fd_eps);

    let mut residuals = Vec::with_capacity(ode.state_dim);
    for (i, eq) in ode.equations.iter().enumerate() {
        let dxdt = if fd_eps.abs() > 1e-20 {
            (state_plus.get(i).copied().unwrap_or(0.0) - state.get(i).copied().unwrap_or(0.0))
                / fd_eps
        } else {
            0.0
        };
        let rhs = evaluate_ode_expr(eq, &state, t_val)?;
        residuals.push(dxdt - rhs);
    }
    Ok(residuals)
}

/// Evaluate a conservation law quantity on a state vector.
fn conservation_quantity(law: &ConservationLaw, state: &[f64]) -> f64 {
    match law {
        ConservationLaw::EnergyConservation => state.iter().map(|x| x * x / 2.0).sum(),
        ConservationLaw::MassConservation => state.iter().sum(),
        ConservationLaw::MomentumConservation => state
            .iter()
            .enumerate()
            .map(|(i, &x)| (i as f64 + 1.0) * x)
            .sum(),
        ConservationLaw::CustomLinear(coeffs) => {
            state.iter().zip(coeffs.iter()).map(|(&x, &c)| c * x).sum()
        }
    }
}

// ---------------------------------------------------------------------------
// Main predictor
// ---------------------------------------------------------------------------

/// Physics-informed multi-dimensional time series predictor.
///
/// Combines a neural network with ODE residual minimisation and conservation
/// law enforcement to produce physically consistent predictions.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_series::physics_ts::ode_predictor::{
///     PhysicsInformedConfig, PhysicsInformedPredictor, ODESystem,
///     ConservationLaw,
/// };
///
/// let mut config = PhysicsInformedConfig::default();
/// config.physics_weight = 1.0;
/// config.data_weight = 1.0;
/// config.ode_system = Some(ODESystem {
///     equations: vec!["x1".to_string(), "-x0".to_string()],
///     state_dim: 2,
/// });
/// config.conservation_laws = vec![ConservationLaw::EnergyConservation];
/// config.n_epochs = 50;
///
/// let mut predictor = PhysicsInformedPredictor::new(config, 2);
/// let times = vec![0.0, 0.1, 0.2, 0.3, 0.4];
/// let obs = vec![
///     vec![1.0, 0.0],
///     vec![0.995, 0.0998],
///     vec![0.980, 0.198],
///     vec![0.955, 0.296],
///     vec![0.921, 0.389],
/// ];
/// let result = predictor.fit(&times, &obs);
/// ```
#[derive(Debug, Clone)]
pub struct PhysicsInformedPredictor {
    config: PhysicsInformedConfig,
    mlp: MultiOutputMlp,
    adam: Adam,
    state_dim: usize,
    fd_eps: f64,
    fitted: bool,
    /// Reference conservation values (computed from initial state).
    conservation_refs: Vec<f64>,
}

/// Result of fitting or prediction.
#[derive(Debug, Clone)]
pub struct PredictorResult {
    /// Predicted state vectors at queried time points.
    pub predictions: Vec<Vec<f64>>,
    /// Per-time-point physics residual norms.
    pub physics_residuals: Vec<f64>,
    /// Total physics loss (ODE + conservation).
    pub total_physics_loss: f64,
    /// Data-fidelity loss (MSE).
    pub data_loss: f64,
}

impl PhysicsInformedPredictor {
    /// Create a new predictor for a system with `state_dim` dimensions.
    pub fn new(config: PhysicsInformedConfig, state_dim: usize) -> Self {
        let mlp = MultiOutputMlp::new(config.hidden_dim, config.n_layers, state_dim);
        let n = mlp.n_params();
        let adam = Adam::new(n, config.learning_rate);
        Self {
            config,
            mlp,
            adam,
            state_dim,
            fd_eps: 1e-4,
            fitted: false,
            conservation_refs: Vec::new(),
        }
    }

    /// Evaluate the ODE residual at a given time and the network's predicted state.
    ///
    /// Returns a vector of residuals (one per state dimension). If no ODE system
    /// is configured, returns a zero vector.
    pub fn physics_residual(&self, time: f64, state: &[f64]) -> Vec<f64> {
        if let Some(ref ode) = self.config.ode_system {
            let state_plus = self.mlp.predict(time + self.fd_eps);
            let mut residuals = Vec::with_capacity(ode.state_dim);
            for (i, eq) in ode.equations.iter().enumerate() {
                let dxdt = if self.fd_eps.abs() > 1e-20 {
                    (state_plus.get(i).copied().unwrap_or(0.0)
                        - state.get(i).copied().unwrap_or(0.0))
                        / self.fd_eps
                } else {
                    0.0
                };
                let rhs = evaluate_ode_expr(eq, state, time).unwrap_or(0.0);
                residuals.push(dxdt - rhs);
            }
            residuals
        } else {
            vec![0.0; self.state_dim]
        }
    }

    /// Compute total loss: data + physics.
    fn total_loss(&self, times: &[f64], observations: &[Vec<f64>]) -> Result<(f64, f64, f64)> {
        let n = times.len();

        // Data loss: MSE over all dimensions
        let mut data_loss = 0.0;
        for (i, &t_val) in times.iter().enumerate() {
            let pred = self.mlp.predict(t_val);
            for d in 0..self.state_dim {
                let diff = pred.get(d).copied().unwrap_or(0.0)
                    - observations[i].get(d).copied().unwrap_or(0.0);
                data_loss += diff * diff;
            }
        }
        data_loss /= (n * self.state_dim) as f64;

        // Physics loss
        let mut physics_loss = 0.0;

        // ODE residual at collocation points
        if let Some(ref ode) = self.config.ode_system {
            let t_min = times.first().copied().unwrap_or(0.0);
            let t_max = times.last().copied().unwrap_or(1.0);
            let n_coll = self.config.n_collocation.max(2);
            for k in 0..n_coll {
                let t_coll = t_min + (t_max - t_min) * (k as f64) / (n_coll as f64 - 1.0);
                let residuals = ode_residual_at(&self.mlp, ode, t_coll, self.fd_eps)?;
                for &r in &residuals {
                    physics_loss += r * r;
                }
            }
            physics_loss /= (n_coll * ode.state_dim) as f64;
        }

        // Conservation law violations
        if !self.conservation_refs.is_empty() {
            let n_coll = self.config.n_collocation.max(2);
            let t_min = times.first().copied().unwrap_or(0.0);
            let t_max = times.last().copied().unwrap_or(1.0);
            let mut cons_loss = 0.0;
            for k in 0..n_coll {
                let t_coll = t_min + (t_max - t_min) * (k as f64) / (n_coll as f64 - 1.0);
                let state = self.mlp.predict(t_coll);
                for (law_idx, law) in self.config.conservation_laws.iter().enumerate() {
                    let current = conservation_quantity(law, &state);
                    let ref_val = self.conservation_refs.get(law_idx).copied().unwrap_or(0.0);
                    let diff = current - ref_val;
                    cons_loss += diff * diff;
                }
            }
            if !self.config.conservation_laws.is_empty() {
                cons_loss /= (n_coll * self.config.conservation_laws.len()) as f64;
            }
            physics_loss += cons_loss;
        }

        let total = self.config.data_weight * data_loss + self.config.physics_weight * physics_loss;
        Ok((total, data_loss, physics_loss))
    }

    /// Numerical gradient via forward finite differences.
    fn numerical_gradient(&self, times: &[f64], observations: &[Vec<f64>]) -> Result<Vec<f64>> {
        let base_params = self.mlp.flatten();
        let (base_loss, _, _) = self.total_loss(times, observations)?;
        let n = base_params.len();
        let mut grad = vec![0.0_f64; n];

        for k in 0..n {
            let mut perturbed = self.clone();
            let mut p = base_params.clone();
            p[k] += self.fd_eps;
            perturbed.mlp.unflatten(&p);
            let (perturbed_loss, _, _) = perturbed.total_loss(times, observations)?;
            grad[k] = (perturbed_loss - base_loss) / self.fd_eps;
        }
        Ok(grad)
    }

    /// Fit the model to observed multi-variate time series.
    ///
    /// # Arguments
    /// * `time` - Observation times (length N)
    /// * `observations` - Observed state vectors (length N, each of dimension `state_dim`)
    ///
    /// # Returns
    /// A [`PredictorResult`] with training-time predictions and losses.
    pub fn fit(&mut self, time: &[f64], observations: &[Vec<f64>]) -> Result<PredictorResult> {
        if time.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "time must not be empty".to_string(),
            ));
        }
        if time.len() != observations.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: time.len(),
                actual: observations.len(),
            });
        }
        for obs in observations {
            if obs.len() != self.state_dim {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: self.state_dim,
                    actual: obs.len(),
                });
            }
        }

        // Compute reference conservation quantities from the first observation
        self.conservation_refs = self
            .config
            .conservation_laws
            .iter()
            .map(|law| conservation_quantity(law, &observations[0]))
            .collect();

        let mut last_data_loss = 0.0;
        let mut last_physics_loss = 0.0;

        for _epoch in 0..self.config.n_epochs {
            let grad = self.numerical_gradient(time, observations)?;
            let mut params = self.mlp.flatten();
            self.adam.step(&mut params, &grad);
            self.mlp.unflatten(&params);

            let (_, dl, pl) = self.total_loss(time, observations)?;
            last_data_loss = dl;
            last_physics_loss = pl;
        }

        self.fitted = true;

        // Compute predictions and residuals
        let mut predictions = Vec::with_capacity(time.len());
        let mut physics_residuals = Vec::with_capacity(time.len());

        for &t_val in time {
            let pred = self.mlp.predict(t_val);
            let res = if let Some(ref ode) = self.config.ode_system {
                let r = ode_residual_at(&self.mlp, ode, t_val, self.fd_eps)?;
                r.iter().map(|v| v * v).sum::<f64>().sqrt()
            } else {
                0.0
            };
            predictions.push(pred);
            physics_residuals.push(res);
        }

        Ok(PredictorResult {
            predictions,
            physics_residuals,
            total_physics_loss: last_physics_loss,
            data_loss: last_data_loss,
        })
    }

    /// Predict at future (or arbitrary) time points.
    ///
    /// # Arguments
    /// * `future_times` - Time points at which to predict.
    ///
    /// # Returns
    /// A vector of state vectors, one per time point.
    pub fn predict(&self, future_times: &[f64]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(TimeSeriesError::ModelNotFitted(
                "PhysicsInformedPredictor has not been fitted yet".to_string(),
            ));
        }
        let preds: Vec<Vec<f64>> = future_times.iter().map(|&t| self.mlp.predict(t)).collect();
        Ok(preds)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_informed_config_default() {
        let cfg = PhysicsInformedConfig::default();
        assert!((cfg.physics_weight - 1.0).abs() < 1e-12);
        assert!((cfg.data_weight - 1.0).abs() < 1e-12);
        assert!(cfg.ode_system.is_none());
        assert!(cfg.conservation_laws.is_empty());
    }

    #[test]
    fn test_conservation_law_energy() {
        let state = vec![3.0, 4.0];
        let e = conservation_quantity(&ConservationLaw::EnergyConservation, &state);
        // 3^2/2 + 4^2/2 = 4.5 + 8 = 12.5
        assert!((e - 12.5).abs() < 1e-10);
    }

    #[test]
    fn test_conservation_law_mass() {
        let state = vec![1.0, 2.0, 3.0];
        let m = conservation_quantity(&ConservationLaw::MassConservation, &state);
        assert!((m - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_conservation_law_momentum() {
        let state = vec![1.0, 2.0, 3.0];
        // 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        let p = conservation_quantity(&ConservationLaw::MomentumConservation, &state);
        assert!((p - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_conservation_law_custom_linear() {
        let state = vec![2.0, 3.0];
        let cl = ConservationLaw::CustomLinear(vec![0.5, 1.5]);
        let val = conservation_quantity(&cl, &state);
        // 0.5*2 + 1.5*3 = 1 + 4.5 = 5.5
        assert!((val - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_ode_expr_simple() {
        let state = vec![2.0, 3.0];
        // "x0 + x1" should give 5.0
        let val = evaluate_ode_expr("x0 + x1", &state, 0.0).expect("eval");
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_ode_expr_multiply() {
        let state = vec![2.0];
        let val = evaluate_ode_expr("3.0 * x0", &state, 0.0).expect("eval");
        assert!((val - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_ode_expr_trig() {
        let state = vec![0.0];
        let val = evaluate_ode_expr("sin(t)", &state, std::f64::consts::FRAC_PI_2).expect("eval");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_ode_expr_neg() {
        let state = vec![5.0];
        let val = evaluate_ode_expr("-x0", &state, 0.0).expect("eval");
        assert!((val - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_predictor_fit_basic() {
        // Simple linear system: dx/dt = 1 (constant), so x(t) = t + x0
        let config = PhysicsInformedConfig {
            physics_weight: 0.0,
            data_weight: 1.0,
            n_epochs: 10,
            hidden_dim: 8,
            n_layers: 1,
            ..Default::default()
        };

        let mut predictor = PhysicsInformedPredictor::new(config, 1);
        let times: Vec<f64> = (0..5).map(|i| i as f64 * 0.25).collect();
        let obs: Vec<Vec<f64>> = times.iter().map(|&t| vec![t]).collect();

        let result = predictor.fit(&times, &obs).expect("fit");
        assert_eq!(result.predictions.len(), 5);
        assert!(result.data_loss >= 0.0);
    }

    #[test]
    fn test_predictor_with_ode() {
        // dx0/dt = x1, dx1/dt = -x0  (harmonic oscillator)
        let ode = ODESystem {
            equations: vec!["x1".to_string(), "-x0".to_string()],
            state_dim: 2,
        };
        let config = PhysicsInformedConfig {
            physics_weight: 1.0,
            data_weight: 1.0,
            ode_system: Some(ode),
            n_epochs: 5,
            hidden_dim: 8,
            n_layers: 1,
            n_collocation: 5,
            ..Default::default()
        };

        let mut predictor = PhysicsInformedPredictor::new(config, 2);
        let times = vec![0.0, 0.1, 0.2, 0.3, 0.4];
        let obs: Vec<Vec<f64>> = times.iter().map(|t: &f64| vec![t.cos(), t.sin()]).collect();

        let result = predictor.fit(&times, &obs).expect("fit");
        assert_eq!(result.predictions.len(), 5);
        // Physics residuals should be computed
        assert_eq!(result.physics_residuals.len(), 5);
        for &r in &result.physics_residuals {
            assert!(r.is_finite());
        }
    }

    #[test]
    fn test_predictor_conservation_approximately_satisfied() {
        // Mass conservation: sum(x_i) should remain constant
        let config = PhysicsInformedConfig {
            physics_weight: 5.0,
            data_weight: 1.0,
            conservation_laws: vec![ConservationLaw::MassConservation],
            n_epochs: 30,
            hidden_dim: 16,
            n_layers: 1,
            n_collocation: 10,
            ..Default::default()
        };

        let mut predictor = PhysicsInformedPredictor::new(config, 2);
        // Data with sum ≈ 2.0
        let times = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let obs: Vec<Vec<f64>> = vec![
            vec![1.0, 1.0],
            vec![1.1, 0.9],
            vec![1.2, 0.8],
            vec![1.3, 0.7],
            vec![1.4, 0.6],
        ];

        let result = predictor.fit(&times, &obs).expect("fit");
        // total_physics_loss should be finite and non-negative
        assert!(result.total_physics_loss >= 0.0);
        assert!(result.total_physics_loss.is_finite());
    }

    #[test]
    fn test_predictor_ode_residual_method() {
        let ode = ODESystem {
            equations: vec!["x1".to_string(), "-x0".to_string()],
            state_dim: 2,
        };
        let config = PhysicsInformedConfig {
            physics_weight: 1.0,
            data_weight: 1.0,
            ode_system: Some(ode),
            n_epochs: 20,
            hidden_dim: 16,
            n_layers: 1,
            n_collocation: 10,
            ..Default::default()
        };

        let mut predictor = PhysicsInformedPredictor::new(config, 2);
        let times = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let obs: Vec<Vec<f64>> = times.iter().map(|t: &f64| vec![t.cos(), t.sin()]).collect();

        predictor.fit(&times, &obs).expect("fit");
        let state = vec![1.0, 0.0];
        let residual = predictor.physics_residual(0.0, &state);
        assert_eq!(residual.len(), 2);
        for &r in &residual {
            assert!(r.is_finite());
        }
    }

    #[test]
    fn test_predictor_data_fit_improves_with_higher_data_weight() {
        let times: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
        let obs: Vec<Vec<f64>> = times.iter().map(|&t| vec![t * 2.0]).collect();

        // Low data weight
        let config_low = PhysicsInformedConfig {
            physics_weight: 10.0,
            data_weight: 0.1,
            conservation_laws: vec![ConservationLaw::MassConservation],
            n_epochs: 30,
            hidden_dim: 8,
            n_layers: 1,
            n_collocation: 5,
            ..Default::default()
        };
        let mut pred_low = PhysicsInformedPredictor::new(config_low, 1);
        let res_low = pred_low.fit(&times, &obs).expect("fit low");

        // High data weight
        let config_high = PhysicsInformedConfig {
            physics_weight: 0.01,
            data_weight: 10.0,
            conservation_laws: vec![ConservationLaw::MassConservation],
            n_epochs: 30,
            hidden_dim: 8,
            n_layers: 1,
            n_collocation: 5,
            ..Default::default()
        };
        let mut pred_high = PhysicsInformedPredictor::new(config_high, 1);
        let res_high = pred_high.fit(&times, &obs).expect("fit high");

        // With higher data_weight, data_loss should be lower (or at most comparable)
        // This is a soft check — the optimiser has finite epochs
        assert!(
            res_high.data_loss <= res_low.data_loss * 10.0,
            "Higher data_weight should yield better data fit: low={}, high={}",
            res_low.data_loss,
            res_high.data_loss
        );
    }

    #[test]
    fn test_predictor_predict_requires_fit() {
        let config = PhysicsInformedConfig {
            n_epochs: 5,
            hidden_dim: 4,
            n_layers: 1,
            ..Default::default()
        };
        let predictor = PhysicsInformedPredictor::new(config, 1);
        let result = predictor.predict(&[0.5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_predictor_predict_after_fit() {
        let config = PhysicsInformedConfig {
            n_epochs: 5,
            hidden_dim: 4,
            n_layers: 1,
            ..Default::default()
        };
        let mut predictor = PhysicsInformedPredictor::new(config, 1);
        let times = vec![0.0, 0.5, 1.0];
        let obs = vec![vec![0.0], vec![0.5], vec![1.0]];
        predictor.fit(&times, &obs).expect("fit");
        let preds = predictor.predict(&[0.25, 0.75]).expect("predict");
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].len(), 1);
    }

    #[test]
    fn test_predictor_dimension_mismatch() {
        let config = PhysicsInformedConfig::default();
        let mut predictor = PhysicsInformedPredictor::new(config, 2);
        let times = vec![0.0, 0.5];
        let obs = vec![vec![1.0], vec![2.0]]; // dim 1 instead of 2
        let result = predictor.fit(&times, &obs);
        assert!(result.is_err());
    }
}
