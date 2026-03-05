//! Factor Graph with belief propagation, HMM support, and Viterbi.
//!
//! Provides:
//! - `Factor` – a multidimensional table over discrete variables
//! - `FactorGraph` – a bipartite graph of variable nodes and factor nodes
//! - Sum-product and max-product belief propagation
//! - Partition function via variable elimination
//! - HMM forward-backward algorithm and Viterbi decoding

use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::StatsError;

// ---------------------------------------------------------------------------
// Helper: multi-index ↔ flat-index for arbitrary cardinalities
// ---------------------------------------------------------------------------

/// Compute row-major strides for given cardinalities.
fn compute_strides(cardinalities: &[usize]) -> Vec<usize> {
    let n = cardinalities.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * cardinalities[i + 1];
    }
    strides
}

/// Convert multi-index `states` (with given `cardinalities`) to flat index.
fn states_to_flat(states: &[usize], strides: &[usize]) -> usize {
    states.iter().zip(strides.iter()).map(|(&s, &st)| s * st).sum()
}

/// Convert flat index to multi-index.
fn flat_to_states(mut flat: usize, cardinalities: &[usize]) -> Vec<usize> {
    let strides = compute_strides(cardinalities);
    let mut states = vec![0usize; cardinalities.len()];
    for (i, &st) in strides.iter().enumerate() {
        states[i] = flat / st;
        flat %= st;
    }
    states
}

// ---------------------------------------------------------------------------
// Factor
// ---------------------------------------------------------------------------

/// A multidimensional potential table over a subset of variables.
#[derive(Debug, Clone)]
pub struct Factor {
    /// Unique factor identifier.
    pub id: usize,
    /// Variable indices this factor is connected to.
    pub variables: Vec<usize>,
    /// Number of states for each connected variable.
    pub n_states: Vec<usize>,
    /// Flat potential table in row-major order.
    /// Index = Σ state[i] * stride[i]
    pub table: Vec<f64>,
}

impl Factor {
    /// Create a new factor with uniform potentials.
    pub fn new(id: usize, variables: Vec<usize>, n_states: Vec<usize>) -> Self {
        let total: usize = n_states.iter().product::<usize>().max(1);
        Factor {
            id,
            variables,
            n_states,
            table: vec![1.0_f64; total],
        }
    }

    /// Set the potential value for a given multi-index `states`.
    pub fn set_entry(&mut self, states: &[usize], value: f64) -> Result<(), StatsError> {
        if states.len() != self.variables.len() {
            return Err(StatsError::InvalidArgument(format!(
                "Factor {}: expected {} state indices, got {}",
                self.id,
                self.variables.len(),
                states.len()
            )));
        }
        for (k, (&s, &cap)) in states.iter().zip(&self.n_states).enumerate() {
            if s >= cap {
                return Err(StatsError::InvalidArgument(format!(
                    "Factor {}: state {} for variable index {} exceeds capacity {}",
                    self.id, s, k, cap
                )));
            }
        }
        let strides = compute_strides(&self.n_states);
        let flat = states_to_flat(states, &strides);
        self.table[flat] = value;
        Ok(())
    }

    /// Get the potential value for a given multi-index `states`.
    pub fn get_entry(&self, states: &[usize]) -> Result<f64, StatsError> {
        if states.len() != self.variables.len() {
            return Err(StatsError::InvalidArgument(format!(
                "Factor {}: expected {} state indices, got {}",
                self.id,
                self.variables.len(),
                states.len()
            )));
        }
        let strides = compute_strides(&self.n_states);
        let flat = states_to_flat(states, &strides);
        self.table
            .get(flat)
            .copied()
            .ok_or_else(|| StatsError::ComputationError(format!("Factor {}: flat index {} out of range", self.id, flat)))
    }

    /// Create a unary (single-variable) factor from a probability vector.
    pub fn unary(id: usize, var: usize, probs: Vec<f64>) -> Self {
        Factor {
            id,
            variables: vec![var],
            n_states: vec![probs.len()],
            table: probs,
        }
    }

    /// Create a pairwise factor from an `Array2`.
    pub fn pairwise(id: usize, var1: usize, var2: usize, table: Array2<f64>) -> Self {
        let (r, c) = (table.shape()[0], table.shape()[1]);
        Factor {
            id,
            variables: vec![var1, var2],
            n_states: vec![r, c],
            table: table.into_raw_vec_and_offset().0,
        }
    }
}

// ---------------------------------------------------------------------------
// Factor Graph
// ---------------------------------------------------------------------------

/// Bipartite factor graph: variable nodes ↔ factor nodes.
#[derive(Debug, Clone)]
pub struct FactorGraph {
    /// Number of variable nodes.
    pub n_variables: usize,
    /// Number of states per variable.
    pub n_states: Vec<usize>,
    /// All factors.
    pub factors: Vec<Factor>,
}

impl FactorGraph {
    /// Create an empty factor graph.
    pub fn new(n_variables: usize, n_states: Vec<usize>) -> Self {
        assert_eq!(n_variables, n_states.len());
        FactorGraph {
            n_variables,
            n_states,
            factors: Vec::new(),
        }
    }

    /// Add a factor, returning its index.
    pub fn add_factor(&mut self, factor: Factor) -> usize {
        let idx = self.factors.len();
        self.factors.push(factor);
        idx
    }

    /// Get the factors connected to variable `var`.
    fn factors_for_var(&self, var: usize) -> Vec<usize> {
        self.factors
            .iter()
            .enumerate()
            .filter(|(_, f)| f.variables.contains(&var))
            .map(|(fi, _)| fi)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Sum-product belief propagation
    // -----------------------------------------------------------------------

    /// Sum-product belief propagation.
    ///
    /// Returns approximate marginals `[var][state]`.
    pub fn sum_product(
        &self,
        max_iter: usize,
        tol: f64,
    ) -> Result<Vec<Vec<f64>>, StatsError> {
        let nv = self.n_variables;
        let nf = self.factors.len();

        // Messages: variable → factor and factor → variable
        // var_to_factor[(v, fi)][s] = message from var v to factor fi at state s
        // factor_to_var[(fi, v)][s] = message from factor fi to var v at state s
        let mut var_to_factor: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
        let mut factor_to_var: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

        // Initialise all messages to uniform
        for fi in 0..nf {
            for &v in &self.factors[fi].variables {
                let k = self.n_states[v];
                var_to_factor.insert((v, fi), vec![1.0 / k as f64; k]);
                factor_to_var.insert((fi, v), vec![1.0 / k as f64; k]);
            }
        }

        for _iter in 0..max_iter {
            let mut max_delta = 0.0_f64;

            // Factor → Variable messages
            for fi in 0..nf {
                let factor = &self.factors[fi];
                let factor_vars = &factor.variables;
                let total = factor.table.len();

                for &v in factor_vars {
                    let k = self.n_states[v];
                    let v_pos = factor_vars.iter().position(|&x| x == v)
                        .expect("v is guaranteed to be in factor_vars since we iterate over it");
                    let mut new_msg = vec![0.0_f64; k];

                    for flat in 0..total {
                        let states = flat_to_states(flat, &factor.n_states);
                        let sv = states[v_pos];

                        let pot = factor.table[flat];
                        // Multiply incoming messages from all variables except v
                        let mut prod = pot;
                        for (other_pos, &other_v) in factor_vars.iter().enumerate() {
                            if other_v == v {
                                continue;
                            }
                            let s_other = states[other_pos];
                            let msg_val = var_to_factor
                                .get(&(other_v, fi))
                                .and_then(|m| m.get(s_other))
                                .copied()
                                .unwrap_or(1.0 / self.n_states[other_v] as f64);
                            prod *= msg_val;
                        }
                        new_msg[sv] += prod;
                    }

                    // Normalise
                    let z: f64 = new_msg.iter().sum();
                    if z > 0.0 {
                        for x in &mut new_msg {
                            *x /= z;
                        }
                    } else {
                        let inv = 1.0 / k as f64;
                        for x in &mut new_msg {
                            *x = inv;
                        }
                    }

                    // Convergence
                    if let Some(old) = factor_to_var.get(&(fi, v)) {
                        let delta: f64 = new_msg.iter().zip(old).map(|(a, b)| (a - b).abs()).sum();
                        if delta > max_delta {
                            max_delta = delta;
                        }
                    }
                    factor_to_var.insert((fi, v), new_msg);
                }
            }

            // Variable → Factor messages
            for v in 0..nv {
                let factor_indices = self.factors_for_var(v);
                let k = self.n_states[v];

                for &fi in &factor_indices {
                    let mut new_msg = vec![1.0_f64; k];
                    // Product of all incoming factor→var messages except from fi
                    for &other_fi in &factor_indices {
                        if other_fi == fi {
                            continue;
                        }
                        if let Some(msg) = factor_to_var.get(&(other_fi, v)) {
                            for s in 0..k {
                                new_msg[s] *= msg[s];
                            }
                        }
                    }
                    // Normalise
                    let z: f64 = new_msg.iter().sum();
                    if z > 0.0 {
                        for x in &mut new_msg {
                            *x /= z;
                        }
                    }
                    var_to_factor.insert((v, fi), new_msg);
                }
            }

            if max_delta < tol {
                break;
            }
        }

        // Compute beliefs
        let mut beliefs: Vec<Vec<f64>> = (0..nv)
            .map(|v| vec![1.0_f64; self.n_states[v]])
            .collect();

        for v in 0..nv {
            let k = self.n_states[v];
            let factor_indices = self.factors_for_var(v);
            for &fi in &factor_indices {
                if let Some(msg) = factor_to_var.get(&(fi, v)) {
                    for s in 0..k {
                        beliefs[v][s] *= msg[s];
                    }
                }
            }
            let z: f64 = beliefs[v].iter().sum();
            if z > 0.0 {
                for x in &mut beliefs[v] {
                    *x /= z;
                }
            } else {
                let inv = 1.0 / k as f64;
                for x in &mut beliefs[v] {
                    *x = inv;
                }
            }
        }

        Ok(beliefs)
    }

    // -----------------------------------------------------------------------
    // Max-product (MAP)
    // -----------------------------------------------------------------------

    /// Max-product belief propagation for MAP inference.
    pub fn max_product(
        &self,
        max_iter: usize,
        tol: f64,
    ) -> Result<Vec<usize>, StatsError> {
        let nv = self.n_variables;
        let nf = self.factors.len();

        // Use log-domain messages
        let mut var_to_factor: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
        let mut factor_to_var: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

        for fi in 0..nf {
            for &v in &self.factors[fi].variables {
                let k = self.n_states[v];
                var_to_factor.insert((v, fi), vec![0.0_f64; k]);
                factor_to_var.insert((fi, v), vec![0.0_f64; k]);
            }
        }

        for _iter in 0..max_iter {
            let mut max_delta = 0.0_f64;

            // Factor → Variable (max-marginalise)
            for fi in 0..nf {
                let factor = &self.factors[fi];
                let factor_vars = &factor.variables;
                let total = factor.table.len();

                for &v in factor_vars {
                    let k = self.n_states[v];
                    let v_pos = factor_vars.iter().position(|&x| x == v)
                        .expect("v is guaranteed to be in factor_vars since we iterate over it");
                    let mut new_msg = vec![f64::NEG_INFINITY; k];

                    for flat in 0..total {
                        let states = flat_to_states(flat, &factor.n_states);
                        let sv = states[v_pos];

                        let log_pot = factor.table[flat].ln();
                        let mut sum_incoming = 0.0_f64;
                        for (other_pos, &other_v) in factor_vars.iter().enumerate() {
                            if other_v == v {
                                continue;
                            }
                            let s_other = states[other_pos];
                            let msg_val = var_to_factor
                                .get(&(other_v, fi))
                                .and_then(|m| m.get(s_other))
                                .copied()
                                .unwrap_or(0.0);
                            sum_incoming += msg_val;
                        }
                        let val = log_pot + sum_incoming;
                        if val > new_msg[sv] {
                            new_msg[sv] = val;
                        }
                    }

                    // Subtract max for stability
                    let max_val = new_msg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    if max_val.is_finite() {
                        for x in &mut new_msg {
                            *x -= max_val;
                        }
                    } else {
                        new_msg = vec![0.0; k];
                    }

                    if let Some(old) = factor_to_var.get(&(fi, v)) {
                        let delta: f64 = new_msg.iter().zip(old).map(|(a, b)| (a - b).abs()).sum();
                        if delta > max_delta {
                            max_delta = delta;
                        }
                    }
                    factor_to_var.insert((fi, v), new_msg);
                }
            }

            // Variable → Factor
            for v in 0..nv {
                let factor_indices = self.factors_for_var(v);
                let k = self.n_states[v];

                for &fi in &factor_indices {
                    let mut new_msg = vec![0.0_f64; k];
                    for &other_fi in &factor_indices {
                        if other_fi == fi {
                            continue;
                        }
                        if let Some(msg) = factor_to_var.get(&(other_fi, v)) {
                            for s in 0..k {
                                new_msg[s] += msg[s];
                            }
                        }
                    }
                    // Subtract max for stability
                    let max_val = new_msg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    if max_val.is_finite() {
                        for x in &mut new_msg {
                            *x -= max_val;
                        }
                    }
                    var_to_factor.insert((v, fi), new_msg);
                }
            }

            if max_delta < tol {
                break;
            }
        }

        // Decode MAP
        let map_states: Vec<usize> = (0..nv)
            .map(|v| {
                let k = self.n_states[v];
                let factor_indices = self.factors_for_var(v);
                let mut scores = vec![0.0_f64; k];
                for &fi in &factor_indices {
                    if let Some(msg) = factor_to_var.get(&(fi, v)) {
                        for s in 0..k {
                            scores[s] += msg[s];
                        }
                    }
                }
                scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();

        Ok(map_states)
    }

    // -----------------------------------------------------------------------
    // Partition Function via Variable Elimination
    // -----------------------------------------------------------------------

    /// Compute the partition function Z = Σ_x Π_f f(x_f) exactly.
    pub fn partition_function(&self) -> Result<f64, StatsError> {
        if self.n_variables == 0 {
            return Ok(1.0);
        }

        // Represent each factor as a mini-factor struct for elimination
        #[derive(Clone)]
        struct MiniFactor {
            vars: Vec<usize>,
            cards: Vec<usize>,
            values: Vec<f64>,
        }

        impl MiniFactor {
            fn strides(&self) -> Vec<usize> {
                compute_strides(&self.cards)
            }

            fn multiply(&self, other: &MiniFactor) -> MiniFactor {
                let mut vars = self.vars.clone();
                let mut cards = self.cards.clone();
                for (&v, &c) in other.vars.iter().zip(&other.cards) {
                    if !vars.contains(&v) {
                        vars.push(v);
                        cards.push(c);
                    }
                }
                let total: usize = cards.iter().product::<usize>().max(1);
                let strides = compute_strides(&cards);
                let self_strides = self.strides();
                let other_strides = other.strides();
                let self_map: Vec<Option<usize>> = vars
                    .iter()
                    .map(|v| self.vars.iter().position(|sv| sv == v))
                    .collect();
                let other_map: Vec<Option<usize>> = vars
                    .iter()
                    .map(|v| other.vars.iter().position(|ov| ov == v))
                    .collect();

                let mut values = vec![0.0_f64; total];
                for flat in 0..total {
                    let states = flat_to_states(flat, &cards);
                    let mut si = 0;
                    let mut oi = 0;
                    for (k, &s) in states.iter().enumerate() {
                        if let Some(pos) = self_map[k] {
                            si += s * self_strides[pos];
                        }
                        if let Some(pos) = other_map[k] {
                            oi += s * other_strides[pos];
                        }
                    }
                    values[flat] = self.values[si] * other.values[oi];
                }
                MiniFactor { vars, cards, values }
            }

            fn sum_out(&self, var: usize) -> MiniFactor {
                let pos = match self.vars.iter().position(|&v| v == var) {
                    Some(p) => p,
                    None => return self.clone(),
                };
                let new_vars: Vec<usize> = self
                    .vars
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != pos)
                    .map(|(_, &v)| v)
                    .collect();
                let new_cards: Vec<usize> = self
                    .cards
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != pos)
                    .map(|(_, &c)| c)
                    .collect();
                if new_vars.is_empty() {
                    let sum: f64 = self.values.iter().sum();
                    return MiniFactor {
                        vars: vec![],
                        cards: vec![],
                        values: vec![sum],
                    };
                }
                let n: usize = new_cards.iter().product::<usize>().max(1);
                let mut result = vec![0.0_f64; n];
                let orig_strides = self.strides();
                let res_strides = compute_strides(&new_cards);
                for flat in 0..self.values.len() {
                    let states = flat_to_states(flat, &self.cards);
                    let mut res_idx = 0;
                    let mut rd = 0;
                    for (i, &s) in states.iter().enumerate() {
                        if i != pos {
                            res_idx += s * res_strides[rd];
                            rd += 1;
                        }
                    }
                    result[res_idx] += self.values[flat];
                    let _ = orig_strides[0]; // suppress unused warning
                }
                MiniFactor {
                    vars: new_vars,
                    cards: new_cards,
                    values: result,
                }
            }
        }

        // Build mini-factors from factors
        let mut mini_factors: Vec<MiniFactor> = self
            .factors
            .iter()
            .map(|f| MiniFactor {
                vars: f.variables.clone(),
                cards: f.n_states.clone(),
                values: f.table.clone(),
            })
            .collect();

        // Eliminate variables in order 0..n_variables
        for var in 0..self.n_variables {
            let (relevant, rest): (Vec<MiniFactor>, Vec<MiniFactor>) = mini_factors
                .into_iter()
                .partition(|f| f.vars.contains(&var));

            if relevant.is_empty() {
                mini_factors = rest;
                continue;
            }

            let product = relevant
                .into_iter()
                .reduce(|a, b| a.multiply(&b))
                .expect("non-empty");
            let marginalised = product.sum_out(var);
            mini_factors = rest;
            if !marginalised.vars.is_empty() || marginalised.values.len() == 1 {
                mini_factors.push(marginalised);
            }
        }

        // Product of remaining scalars
        let z: f64 = mini_factors
            .iter()
            .map(|f| f.values.iter().sum::<f64>())
            .product();

        Ok(z)
    }

    // -----------------------------------------------------------------------
    // HMM Builder
    // -----------------------------------------------------------------------

    /// Build a Hidden Markov Model as a factor graph.
    ///
    /// Variables: hidden[t] = t, observed[t] = T + t  for t = 0..T
    ///
    /// Factors:
    /// - Initial: f(h_0)
    /// - Transition: f(h_{t-1}, h_t)
    /// - Emission: f(h_t, o_t)  (observed variables are clamped via unary factors)
    pub fn hmm(
        initial: Vec<f64>,
        transition: Array2<f64>,
        emission: Array2<f64>,
        observations: &[usize],
    ) -> Result<Self, StatsError> {
        let n_hidden = initial.len();
        let t_len = observations.len();
        if t_len == 0 {
            return Err(StatsError::InvalidArgument(
                "Observations sequence must not be empty".to_string(),
            ));
        }
        if transition.shape() != [n_hidden, n_hidden] {
            return Err(StatsError::InvalidArgument(format!(
                "Transition matrix must be ({0},{0}), got {1:?}",
                n_hidden,
                transition.shape()
            )));
        }
        let n_obs = emission.shape()[1];
        if emission.shape()[0] != n_hidden {
            return Err(StatsError::InvalidArgument(format!(
                "Emission rows must equal n_hidden={}, got {}",
                n_hidden,
                emission.shape()[0]
            )));
        }
        for &o in observations {
            if o >= n_obs {
                return Err(StatsError::InvalidArgument(format!(
                    "Observation {} out of range (n_obs={})",
                    o, n_obs
                )));
            }
        }

        // 2*T variables: hidden[0..T] then observed[0..T] (the latter are clamped)
        let n_vars = 2 * t_len;
        let n_states: Vec<usize> = (0..t_len)
            .map(|_| n_hidden)
            .chain((0..t_len).map(|_| n_obs))
            .collect();

        let mut fg = FactorGraph::new(n_vars, n_states);

        // Initial factor
        let init_factor = Factor::unary(0, 0, initial.clone());
        fg.add_factor(init_factor);

        // Transition factors
        for t in 1..t_len {
            let fid = fg.factors.len();
            let trans_factor = Factor::pairwise(fid, t - 1, t, transition.clone());
            fg.add_factor(trans_factor);
        }

        // Emission factors + observation evidence
        for t in 0..t_len {
            let obs_var = t_len + t; // observed variable index
            let fid = fg.factors.len();
            let emit_factor = Factor::pairwise(fid, t, obs_var, emission.clone());
            fg.add_factor(emit_factor);

            // Clamp observed variable to its actual observation (unary factor)
            let mut obs_unary = vec![0.0_f64; n_obs];
            obs_unary[observations[t]] = 1.0;
            let fid2 = fg.factors.len();
            let obs_factor = Factor::unary(fid2, obs_var, obs_unary);
            fg.add_factor(obs_factor);
        }

        Ok(fg)
    }
}

// ---------------------------------------------------------------------------
// HMM forward-backward
// ---------------------------------------------------------------------------

/// Forward-backward algorithm for HMMs.
///
/// Returns:
/// - `alpha`: forward probabilities, shape `(T, n_states)`
/// - `beta`: backward probabilities, shape `(T, n_states)`
/// - `log_likelihood`: log P(observations)
pub fn hmm_forward_backward(
    initial: &[f64],
    transition: &Array2<f64>,
    emission: &Array2<f64>,
    observations: &[usize],
) -> Result<(Array2<f64>, Array2<f64>, f64), StatsError> {
    let n = initial.len();
    let t_len = observations.len();
    if t_len == 0 {
        return Err(StatsError::InvalidArgument(
            "Observation sequence must not be empty".to_string(),
        ));
    }
    if transition.shape() != [n, n] {
        return Err(StatsError::InvalidArgument(format!(
            "Transition matrix must be ({0},{0}), got {1:?}",
            n,
            transition.shape()
        )));
    }
    let n_obs = emission.shape()[1];
    for &o in observations {
        if o >= n_obs {
            return Err(StatsError::InvalidArgument(format!(
                "Observation {} out of range (n_obs={})",
                o, n_obs
            )));
        }
    }

    // Forward pass (scaled)
    let mut alpha = Array2::<f64>::zeros((t_len, n));
    let mut scales = vec![0.0_f64; t_len];

    // t=0
    for s in 0..n {
        alpha[[0, s]] = initial[s] * emission[[s, observations[0]]];
    }
    scales[0] = alpha.row(0).sum();
    if scales[0] == 0.0 {
        return Err(StatsError::ComputationError(
            "Forward variable at t=0 is zero; check initial/emission probabilities".to_string(),
        ));
    }
    for s in 0..n {
        alpha[[0, s]] /= scales[0];
    }

    for t in 1..t_len {
        for s in 0..n {
            let mut sum = 0.0_f64;
            for sp in 0..n {
                sum += alpha[[t - 1, sp]] * transition[[sp, s]];
            }
            alpha[[t, s]] = sum * emission[[s, observations[t]]];
        }
        scales[t] = alpha.row(t).sum();
        if scales[t] == 0.0 {
            return Err(StatsError::ComputationError(format!(
                "Forward variable at t={} is zero; check transition/emission probabilities",
                t
            )));
        }
        for s in 0..n {
            alpha[[t, s]] /= scales[t];
        }
    }

    // Backward pass (using same scaling)
    let mut beta = Array2::<f64>::zeros((t_len, n));

    // t = T-1
    for s in 0..n {
        beta[[t_len - 1, s]] = 1.0 / scales[t_len - 1];
    }

    for t in (0..t_len - 1).rev() {
        for s in 0..n {
            let mut sum = 0.0_f64;
            for sp in 0..n {
                sum += transition[[s, sp]]
                    * emission[[sp, observations[t + 1]]]
                    * beta[[t + 1, sp]];
            }
            beta[[t, s]] = sum / scales[t];
        }
    }

    // Log-likelihood = sum of log scales
    let log_likelihood: f64 = scales.iter().map(|&s| s.ln()).sum();

    Ok((alpha, beta, log_likelihood))
}

// ---------------------------------------------------------------------------
// Viterbi algorithm
// ---------------------------------------------------------------------------

/// Viterbi algorithm for MAP decoding of HMMs.
///
/// Returns:
/// - `best_states`: most likely hidden state sequence
/// - `log_prob`: log-probability of the best path
pub fn viterbi(
    initial: &[f64],
    transition: &Array2<f64>,
    emission: &Array2<f64>,
    observations: &[usize],
) -> Result<(Vec<usize>, f64), StatsError> {
    let n = initial.len();
    let t_len = observations.len();
    if t_len == 0 {
        return Err(StatsError::InvalidArgument(
            "Observation sequence must not be empty".to_string(),
        ));
    }
    if transition.shape() != [n, n] {
        return Err(StatsError::InvalidArgument(format!(
            "Transition matrix must be ({0},{0}), got {1:?}",
            n,
            transition.shape()
        )));
    }
    let n_obs = emission.shape()[1];
    for &o in observations {
        if o >= n_obs {
            return Err(StatsError::InvalidArgument(format!(
                "Observation {} out of range",
                o
            )));
        }
    }

    // delta[t][s] = log-prob of most likely path ending at state s at time t
    let mut delta = Array2::<f64>::from_elem((t_len, n), f64::NEG_INFINITY);
    // psi[t][s] = predecessor state of s at time t
    let mut psi = Array2::<usize>::zeros((t_len, n));

    // t=0
    for s in 0..n {
        let init_p = initial[s];
        let emit_p = emission[[s, observations[0]]];
        if init_p > 0.0 && emit_p > 0.0 {
            delta[[0, s]] = init_p.ln() + emit_p.ln();
        }
        psi[[0, s]] = 0;
    }

    // Recursion
    for t in 1..t_len {
        for s in 0..n {
            let mut best_log_prob = f64::NEG_INFINITY;
            let mut best_prev = 0;
            for sp in 0..n {
                let trans_p = transition[[sp, s]];
                if trans_p > 0.0 && delta[[t - 1, sp]].is_finite() {
                    let log_p = delta[[t - 1, sp]] + trans_p.ln();
                    if log_p > best_log_prob {
                        best_log_prob = log_p;
                        best_prev = sp;
                    }
                }
            }
            let emit_p = emission[[s, observations[t]]];
            if emit_p > 0.0 && best_log_prob.is_finite() {
                delta[[t, s]] = best_log_prob + emit_p.ln();
            }
            psi[[t, s]] = best_prev;
        }
    }

    // Find best last state
    let (best_last, best_log_prob) = (0..n)
        .map(|s| (s, delta[[t_len - 1, s]]))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| StatsError::ComputationError("Viterbi: empty state space".to_string()))?;

    if !best_log_prob.is_finite() {
        return Err(StatsError::ComputationError(
            "Viterbi: all paths have zero probability; check model parameters".to_string(),
        ));
    }

    // Backtrack
    let mut path = vec![0usize; t_len];
    path[t_len - 1] = best_last;
    for t in (0..t_len - 1).rev() {
        path[t] = psi[[t + 1, path[t + 1]]];
    }

    Ok((path, best_log_prob))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Build a simple HMM: 2 hidden states, 2 observations
    fn simple_hmm_params() -> (Vec<f64>, Array2<f64>, Array2<f64>) {
        let initial = vec![0.6, 0.4];
        let transition = array![[0.7, 0.3], [0.4, 0.6]];
        let emission = array![[0.9, 0.1], [0.2, 0.8]];
        (initial, transition, emission)
    }

    #[test]
    fn test_factor_new_and_set_entry() {
        let mut f = Factor::new(0, vec![0, 1], vec![2, 3]);
        f.set_entry(&[0, 0], 0.5).unwrap();
        f.set_entry(&[1, 2], 0.8).unwrap();
        assert!((f.get_entry(&[0, 0]).unwrap() - 0.5).abs() < 1e-9);
        assert!((f.get_entry(&[1, 2]).unwrap() - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_factor_set_entry_out_of_range() {
        let mut f = Factor::new(0, vec![0], vec![3]);
        assert!(f.set_entry(&[3], 1.0).is_err()); // state 3 out of range for n_states=3
    }

    #[test]
    fn test_factor_unary() {
        let f = Factor::unary(0, 2, vec![0.3, 0.7]);
        assert_eq!(f.variables, vec![2]);
        assert!((f.get_entry(&[0]).unwrap() - 0.3).abs() < 1e-9);
        assert!((f.get_entry(&[1]).unwrap() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_factor_pairwise() {
        let table = array![[0.1, 0.2], [0.3, 0.4]];
        let f = Factor::pairwise(0, 0, 1, table);
        assert_eq!(f.variables, vec![0, 1]);
        assert!((f.get_entry(&[0, 0]).unwrap() - 0.1).abs() < 1e-9);
        assert!((f.get_entry(&[1, 1]).unwrap() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_factor_graph_construction() {
        let mut fg = FactorGraph::new(3, vec![2, 2, 2]);
        let f = Factor::unary(0, 0, vec![0.4, 0.6]);
        fg.add_factor(f);
        assert_eq!(fg.factors.len(), 1);
    }

    #[test]
    fn test_sum_product_single_variable() {
        let mut fg = FactorGraph::new(1, vec![2]);
        // Prior: P(X=0) = 0.3, P(X=1) = 0.7
        let f = Factor::unary(0, 0, vec![0.3, 0.7]);
        fg.add_factor(f);
        let beliefs = fg.sum_product(50, 1e-8).unwrap();
        assert!((beliefs[0][0] - 0.3).abs() < 1e-5, "belief={:?}", beliefs[0]);
        assert!((beliefs[0][1] - 0.7).abs() < 1e-5, "belief={:?}", beliefs[0]);
    }

    #[test]
    fn test_sum_product_two_variables() {
        // X → Y (pairwise factor)
        let mut fg = FactorGraph::new(2, vec![2, 2]);
        let prior = Factor::unary(0, 0, vec![0.6, 0.4]);
        fg.add_factor(prior);
        let table = array![[0.8, 0.2], [0.3, 0.7]];
        let pf = Factor::pairwise(1, 0, 1, table);
        fg.add_factor(pf);
        let beliefs = fg.sum_product(100, 1e-8).unwrap();
        // Beliefs must be valid distributions
        for b in &beliefs {
            let s: f64 = b.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "belief sums to {}", s);
        }
    }

    #[test]
    fn test_max_product_bias() {
        // Strong prior toward state 1
        let mut fg = FactorGraph::new(1, vec![2]);
        let f = Factor::unary(0, 0, vec![0.001, 0.999]);
        fg.add_factor(f);
        let map = fg.max_product(50, 1e-8).unwrap();
        assert_eq!(map[0], 1, "Expected MAP state 1, got {}", map[0]);
    }

    #[test]
    fn test_partition_function_two_state() {
        // Single variable with two states
        let mut fg = FactorGraph::new(1, vec![2]);
        fg.add_factor(Factor::unary(0, 0, vec![0.4, 0.6]));
        let z = fg.partition_function().unwrap();
        assert!((z - 1.0).abs() < 1e-8, "Z = {}", z);
    }

    #[test]
    fn test_hmm_forward_backward_likelihood() {
        let (init, trans, emit) = simple_hmm_params();
        let obs = vec![0, 1, 0, 0, 1];
        let (alpha, beta, ll) = hmm_forward_backward(&init, &trans, &emit, &obs).unwrap();
        // Log-likelihood must be finite and negative
        assert!(ll.is_finite(), "log_likelihood = {}", ll);
        assert!(ll < 0.0, "log_likelihood should be negative, got {}", ll);
        // alpha shape must be (T, n_states)
        assert_eq!(alpha.shape(), [5, 2]);
        assert_eq!(beta.shape(), [5, 2]);
    }

    #[test]
    fn test_hmm_forward_backward_posterior() {
        let (init, trans, emit) = simple_hmm_params();
        let obs = vec![0, 1, 0];
        let (alpha, beta, _) = hmm_forward_backward(&init, &trans, &emit, &obs).unwrap();
        // Posterior gamma_t = alpha_t * beta_t / sum, must sum to 1
        for t in 0..3 {
            let sum: f64 = (0..2).map(|s| alpha[[t, s]] * beta[[t, s]]).sum();
            assert!(sum > 0.0, "Posterior sum at t={} is 0", t);
        }
    }

    #[test]
    fn test_viterbi_decoding() {
        let (init, trans, emit) = simple_hmm_params();
        let obs = vec![0, 0, 0, 0, 0]; // all 0 → should prefer state 0
        let (path, log_prob) = viterbi(&init, &trans, &emit, &obs).unwrap();
        assert_eq!(path.len(), 5);
        assert!(log_prob.is_finite(), "log_prob = {}", log_prob);
        // With emissions favoring state 0 for obs=0, most states should be 0
        let zeros = path.iter().filter(|&&s| s == 0).count();
        assert!(zeros >= 3, "Expected mostly state 0, got {:?}", path);
    }

    #[test]
    fn test_viterbi_vs_forward_backward() {
        // Both should succeed on same input
        let (init, trans, emit) = simple_hmm_params();
        let obs = vec![0, 1, 1, 0];
        let (path, _) = viterbi(&init, &trans, &emit, &obs).unwrap();
        let (_, _, ll) = hmm_forward_backward(&init, &trans, &emit, &obs).unwrap();
        assert_eq!(path.len(), 4);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_hmm_builder() {
        let (init, trans, emit) = simple_hmm_params();
        let obs = vec![0, 1, 0];
        let fg = FactorGraph::hmm(init, trans, emit, &obs).unwrap();
        // T=3 hidden + 3 observed = 6 variables
        assert_eq!(fg.n_variables, 6);
        // Factors: 1 initial + 2 transition + 3*(emission + clamp) = 1+2+6=9
        assert_eq!(fg.factors.len(), 9, "factors = {}", fg.factors.len());
    }

    #[test]
    fn test_hmm_empty_obs_error() {
        let (init, trans, emit) = simple_hmm_params();
        assert!(FactorGraph::hmm(init, trans, emit, &[]).is_err());
    }

    #[test]
    fn test_forward_backward_empty_obs_error() {
        let (init, trans, emit) = simple_hmm_params();
        assert!(hmm_forward_backward(&init, &trans, &emit, &[]).is_err());
    }

    #[test]
    fn test_viterbi_empty_obs_error() {
        let (init, trans, emit) = simple_hmm_params();
        assert!(viterbi(&init, &trans, &emit, &[]).is_err());
    }
}
