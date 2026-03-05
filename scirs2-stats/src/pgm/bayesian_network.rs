//! Bayesian Network: directed acyclic graphical model with exact/approximate inference.
//!
//! Supports:
//! - Construction with CPTs (conditional probability tables)
//! - DAG validation and topological ordering
//! - Variable elimination for exact marginal inference
//! - Ancestral sampling
//! - Markov blanket computation
//! - d-separation queries

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::StatsError;

// ---------------------------------------------------------------------------
// Conditional Probability Table
// ---------------------------------------------------------------------------

/// Conditional Probability Table (CPT) for a discrete random variable.
///
/// `table[parent_config_index][state]` = P(node=state | parent configuration)
///
/// The parent configuration index is computed as the mixed-radix number:
/// `sum_k parent_state[k] * prod_{j>k} parent_states[j]`
#[derive(Debug, Clone)]
pub struct ConditionalProbability {
    /// Name of the variable this CPT belongs to.
    pub node: String,
    /// Names of parent variables.
    pub parents: Vec<String>,
    /// Number of discrete states for this node.
    pub num_states: usize,
    /// Number of states for each parent (in the same order as `parents`).
    pub parent_states: Vec<usize>,
    /// `table[parent_config][state]` – must sum to 1.0 for each row.
    pub table: Vec<Vec<f64>>,
}

impl ConditionalProbability {
    /// Create a root-node CPT (no parents).  All rows default to uniform.
    pub fn new(node: &str, num_states: usize) -> Self {
        let uniform = vec![1.0 / num_states as f64; num_states];
        ConditionalProbability {
            node: node.to_string(),
            parents: vec![],
            num_states,
            parent_states: vec![],
            table: vec![uniform],
        }
    }

    /// Create a CPT with parents.  All rows default to uniform.
    pub fn with_parents(
        node: &str,
        num_states: usize,
        parents: Vec<String>,
        parent_states: Vec<usize>,
    ) -> Self {
        let n_rows: usize = parent_states.iter().product::<usize>().max(1);
        let uniform = vec![1.0 / num_states as f64; num_states];
        ConditionalProbability {
            node: node.to_string(),
            parents,
            num_states,
            parent_states,
            table: vec![uniform; n_rows],
        }
    }

    /// Compute the row index for a given parent configuration.
    fn config_index(&self, parent_config: &[usize]) -> Result<usize, StatsError> {
        if parent_config.len() != self.parents.len() {
            return Err(StatsError::InvalidArgument(format!(
                "CPT for '{}': expected {} parent values, got {}",
                self.node,
                self.parents.len(),
                parent_config.len()
            )));
        }
        if parent_config.is_empty() {
            return Ok(0);
        }
        let mut idx = 0usize;
        let mut stride = 1usize;
        for k in (0..parent_config.len()).rev() {
            if parent_config[k] >= self.parent_states[k] {
                return Err(StatsError::InvalidArgument(format!(
                    "Parent '{}' state {} out of range (max {})",
                    self.parents[k], parent_config[k], self.parent_states[k] - 1
                )));
            }
            idx += parent_config[k] * stride;
            stride *= self.parent_states[k];
        }
        Ok(idx)
    }

    /// Get P(node = `state` | parent configuration `parent_config`).
    pub fn get_probability(
        &self,
        state: usize,
        parent_config: &[usize],
    ) -> Result<f64, StatsError> {
        if state >= self.num_states {
            return Err(StatsError::InvalidArgument(format!(
                "State {} out of range for node '{}' (max {})",
                state,
                self.node,
                self.num_states - 1
            )));
        }
        let idx = self.config_index(parent_config)?;
        Ok(self.table[idx][state])
    }

    /// Set the probability row for the given parent configuration.
    pub fn set_row(&mut self, parent_config: &[usize], probs: Vec<f64>) -> Result<(), StatsError> {
        if probs.len() != self.num_states {
            return Err(StatsError::InvalidArgument(format!(
                "Expected {} probabilities, got {}",
                self.num_states,
                probs.len()
            )));
        }
        let sum: f64 = probs.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(StatsError::InvalidArgument(format!(
                "Probabilities for node '{}' must sum to 1.0, got {:.6}",
                self.node, sum
            )));
        }
        if probs.iter().any(|&p| p < 0.0) {
            return Err(StatsError::InvalidArgument(format!(
                "Negative probability encountered for node '{}'",
                self.node
            )));
        }
        let idx = self.config_index(parent_config)?;
        self.table[idx] = probs;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Factor – intermediate object used by variable elimination
// ---------------------------------------------------------------------------

/// An unnormalised factor over a set of discrete variables.
#[derive(Debug, Clone)]
struct Factor {
    vars: Vec<String>,
    cardinalities: Vec<usize>,
    /// Row-major table: `values[flat_index]`
    values: Vec<f64>,
}

impl Factor {
    fn new(vars: Vec<String>, cardinalities: Vec<usize>) -> Self {
        let n: usize = cardinalities.iter().product::<usize>().max(1);
        Factor {
            vars,
            cardinalities,
            values: vec![1.0; n],
        }
    }

    /// Compute strides for row-major indexing.
    fn strides(&self) -> Vec<usize> {
        let n = self.cardinalities.len();
        let mut strides = vec![1usize; n];
        for i in (0..n.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.cardinalities[i + 1];
        }
        strides
    }

    /// Multiply two factors, producing a factor over the union of their variables.
    fn multiply(&self, other: &Factor) -> Factor {
        // Union of variables
        let mut vars = self.vars.clone();
        let mut cards = self.cardinalities.clone();
        for (v, &c) in other.vars.iter().zip(&other.cardinalities) {
            if !vars.contains(v) {
                vars.push(v.clone());
                cards.push(c);
            }
        }
        let n: usize = cards.iter().product::<usize>().max(1);

        let result_strides = {
            let len = cards.len();
            let mut s = vec![1usize; len];
            for i in (0..len.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * cards[i + 1];
            }
            s
        };

        let self_map: Vec<Option<usize>> = vars
            .iter()
            .map(|v| self.vars.iter().position(|sv| sv == v))
            .collect();
        let other_map: Vec<Option<usize>> = vars
            .iter()
            .map(|v| other.vars.iter().position(|ov| ov == v))
            .collect();

        let self_strides = self.strides();
        let other_strides = other.strides();

        let mut result_values = vec![0.0_f64; n];
        for flat in 0..n {
            // Decode multi-index in result
            let mut tmp = flat;
            let mut states = vec![0usize; vars.len()];
            for (i, (&stride, _)) in result_strides.iter().zip(vars.iter()).enumerate() {
                states[i] = tmp / stride;
                tmp %= stride;
            }
            // Encode index in self and other
            let mut self_idx = 0usize;
            let mut other_idx = 0usize;
            for (i, &state) in states.iter().enumerate() {
                if let Some(si) = self_map[i] {
                    self_idx += state * self_strides[si];
                }
                if let Some(oi) = other_map[i] {
                    other_idx += state * other_strides[oi];
                }
            }
            result_values[flat] = self.values[self_idx] * other.values[other_idx];
        }
        Factor {
            vars,
            cardinalities: cards,
            values: result_values,
        }
    }

    /// Marginalise (sum out) the variable with name `var_name`.
    fn marginalize(&self, var_name: &str) -> Option<Factor> {
        let pos = self.vars.iter().position(|v| v == var_name)?;
        let new_vars: Vec<String> = self
            .vars
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, v)| v.clone())
            .collect();
        let new_cards: Vec<usize> = self
            .cardinalities
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &c)| c)
            .collect();

        if new_vars.is_empty() {
            let sum: f64 = self.values.iter().sum();
            return Some(Factor {
                vars: vec![],
                cardinalities: vec![],
                values: vec![sum],
            });
        }

        let n: usize = new_cards.iter().product::<usize>().max(1);
        let mut result_vals = vec![0.0_f64; n];

        let orig_strides = self.strides();
        let res_strides: Vec<usize> = {
            let len = new_cards.len();
            let mut s = vec![1usize; len];
            for i in (0..len.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * new_cards[i + 1];
            }
            s
        };

        let total = self.values.len();
        for flat in 0..total {
            // Decode multi-index in original factor
            let mut tmp = flat;
            let mut states = vec![0usize; self.vars.len()];
            for (i, &stride) in orig_strides.iter().enumerate() {
                states[i] = tmp / stride;
                tmp %= stride;
            }
            // Encode index in result (skip pos)
            let mut res_idx = 0usize;
            let mut res_dim = 0usize;
            for (i, &state) in states.iter().enumerate() {
                if i != pos {
                    if res_dim < res_strides.len() {
                        res_idx += state * res_strides[res_dim];
                    }
                    res_dim += 1;
                }
            }
            result_vals[res_idx] += self.values[flat];
        }

        Some(Factor {
            vars: new_vars,
            cardinalities: new_cards,
            values: result_vals,
        })
    }

    /// Restrict (observe) `var_name = state`.
    fn observe(&self, var_name: &str, state: usize) -> Option<Factor> {
        let pos = self.vars.iter().position(|v| v == var_name)?;
        let new_vars: Vec<String> = self
            .vars
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, v)| v.clone())
            .collect();
        let new_cards: Vec<usize> = self
            .cardinalities
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &c)| c)
            .collect();

        let orig_strides = self.strides();
        let n: usize = new_cards.iter().product::<usize>().max(1);
        let mut result_vals = vec![0.0_f64; n];

        let res_strides: Vec<usize> = {
            let len = new_cards.len();
            let mut s = vec![1usize; len];
            for i in (0..len.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * new_cards[i + 1];
            }
            s
        };

        let total = self.values.len();
        for flat in 0..total {
            let mut tmp = flat;
            let mut states = vec![0usize; self.vars.len()];
            for (i, &stride) in orig_strides.iter().enumerate() {
                states[i] = tmp / stride;
                tmp %= stride;
            }
            if states[pos] != state {
                continue;
            }
            let mut res_idx = 0usize;
            let mut res_dim = 0usize;
            for (i, &st) in states.iter().enumerate() {
                if i != pos {
                    if res_dim < res_strides.len() {
                        res_idx += st * res_strides[res_dim];
                    }
                    res_dim += 1;
                }
            }
            result_vals[res_idx] += self.values[flat];
        }

        Some(Factor {
            vars: new_vars,
            cardinalities: new_cards,
            values: result_vals,
        })
    }

    /// Get the value for a single-variable factor at `state`.
    fn value_at(&self, var: &str, state: usize) -> Option<f64> {
        if self.vars.len() == 1 && self.vars[0] == var {
            self.values.get(state).copied()
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Bayesian Network
// ---------------------------------------------------------------------------

/// Directed acyclic graphical model (Bayesian Network).
///
/// Uses discrete variables with conditional probability tables (CPTs).
#[derive(Debug, Clone)]
pub struct BayesianNetwork {
    nodes: Vec<String>,
    /// Number of states for each node (same order as `nodes`).
    node_states: HashMap<String, usize>,
    /// Directed edges parent → child.
    edges: Vec<(String, String)>,
    /// CPTs, keyed by node name.
    cpds: HashMap<String, ConditionalProbability>,
}

impl Default for BayesianNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianNetwork {
    /// Create an empty Bayesian network.
    pub fn new() -> Self {
        BayesianNetwork {
            nodes: Vec::new(),
            node_states: HashMap::new(),
            edges: Vec::new(),
            cpds: HashMap::new(),
        }
    }

    /// Add a discrete node with `num_states` possible values.
    pub fn add_node(&mut self, name: &str, num_states: usize) -> Result<(), StatsError> {
        if num_states == 0 {
            return Err(StatsError::InvalidArgument(
                "num_states must be > 0".to_string(),
            ));
        }
        if self.nodes.contains(&name.to_string()) {
            return Err(StatsError::InvalidArgument(format!(
                "Node '{}' already exists",
                name
            )));
        }
        self.nodes.push(name.to_string());
        self.node_states.insert(name.to_string(), num_states);
        Ok(())
    }

    /// Add a directed edge from `parent` to `child`.
    pub fn add_edge(&mut self, parent: &str, child: &str) -> Result<(), StatsError> {
        if !self.nodes.contains(&parent.to_string()) {
            return Err(StatsError::InvalidArgument(format!(
                "Parent node '{}' not found",
                parent
            )));
        }
        if !self.nodes.contains(&child.to_string()) {
            return Err(StatsError::InvalidArgument(format!(
                "Child node '{}' not found",
                child
            )));
        }
        if parent == child {
            return Err(StatsError::InvalidArgument(
                "Self-loops are not allowed".to_string(),
            ));
        }
        self.edges
            .push((parent.to_string(), child.to_string()));
        if !self.is_dag() {
            self.edges.pop();
            return Err(StatsError::InvalidArgument(format!(
                "Adding edge {}->{} would create a cycle",
                parent, child
            )));
        }
        Ok(())
    }

    /// Add / replace a CPT for a node.
    pub fn add_cpd(&mut self, cpd: ConditionalProbability) -> Result<(), StatsError> {
        if !self.nodes.contains(&cpd.node) {
            return Err(StatsError::InvalidArgument(format!(
                "Node '{}' not found in network",
                cpd.node
            )));
        }
        // Validate all rows sum to 1
        for (row_idx, row) in cpd.table.iter().enumerate() {
            if row.len() != cpd.num_states {
                return Err(StatsError::InvalidArgument(format!(
                    "CPT row {} for '{}' has {} entries, expected {}",
                    row_idx,
                    cpd.node,
                    row.len(),
                    cpd.num_states
                )));
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-5 {
                return Err(StatsError::InvalidArgument(format!(
                    "CPT row {} for '{}' sums to {:.6}, expected 1.0",
                    row_idx, cpd.node, sum
                )));
            }
        }
        self.cpds.insert(cpd.node.clone(), cpd);
        Ok(())
    }

    /// Return parents of a node.
    pub fn parents(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(_, c)| c == node)
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// Return children of a node.
    pub fn children(&self, node: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|(p, _)| p == node)
            .map(|(_, c)| c.clone())
            .collect()
    }

    /// Check whether the graph is a DAG (no directed cycles).
    pub fn is_dag(&self) -> bool {
        // Kahn's algorithm
        let mut in_degree: HashMap<&str, usize> = self
            .nodes
            .iter()
            .map(|n| (n.as_str(), 0))
            .collect();
        for (_, c) in &self.edges {
            *in_degree.entry(c.as_str()).or_insert(0) += 1;
        }
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&n, _)| n)
            .collect();
        let mut count = 0usize;
        while let Some(node) = queue.pop_front() {
            count += 1;
            for (p, c) in &self.edges {
                if p == node {
                    let deg = in_degree.entry(c.as_str()).or_insert(0);
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(c.as_str());
                    }
                }
            }
        }
        count == self.nodes.len()
    }

    /// Topological ordering of nodes.
    pub fn topological_order(&self) -> Result<Vec<String>, StatsError> {
        if !self.is_dag() {
            return Err(StatsError::ComputationError(
                "Graph contains a cycle; topological order is undefined".to_string(),
            ));
        }
        let mut in_degree: HashMap<&str, usize> = self
            .nodes
            .iter()
            .map(|n| (n.as_str(), 0))
            .collect();
        for (_, c) in &self.edges {
            *in_degree.entry(c.as_str()).or_insert(0) += 1;
        }

        // Deterministic order: collect zero-in-degree nodes sorted
        let mut sorted_zero: Vec<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&n, _)| n)
            .collect();
        sorted_zero.sort_unstable();
        let mut queue: VecDeque<&str> = sorted_zero.into_iter().collect();

        let mut result = Vec::new();
        while let Some(node) = queue.pop_front() {
            result.push(node.to_string());
            let mut children: Vec<&str> = self
                .edges
                .iter()
                .filter(|(p, _)| p.as_str() == node)
                .map(|(_, c)| c.as_str())
                .collect();
            children.sort_unstable();
            for c in children {
                let deg = in_degree.entry(c).or_insert(0);
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(c);
                }
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Variable Elimination
    // -----------------------------------------------------------------------

    /// Build the initial factor for node `n` using its CPT.
    fn node_factor(&self, n: &str) -> Result<Factor, StatsError> {
        let cpd = self.cpds.get(n).ok_or_else(|| {
            StatsError::ComputationError(format!("No CPT found for node '{}'", n))
        })?;

        let mut vars = cpd.parents.clone();
        vars.push(n.to_string());

        let mut cards: Vec<usize> = cpd
            .parents
            .iter()
            .map(|p| self.node_states.get(p).copied().unwrap_or(2))
            .collect();
        cards.push(cpd.num_states);

        let total: usize = cards.iter().product::<usize>().max(1);
        let mut values = vec![0.0_f64; total];

        let strides: Vec<usize> = {
            let len = cards.len();
            let mut s = vec![1usize; len];
            for i in (0..len.saturating_sub(1)).rev() {
                s[i] = s[i + 1] * cards[i + 1];
            }
            s
        };

        // Fill values from CPT rows
        let n_parent_configs = cpd.table.len();
        for parent_flat in 0..n_parent_configs {
            // Decode parent configuration
            let mut tmp = parent_flat;
            let mut parent_states_vec = vec![0usize; cpd.parents.len()];

            if cpd.parents.is_empty() {
                // No parents → single row
            } else {
                // CPT row strides
                let parent_stride: Vec<usize> = {
                    let plen = cpd.parents.len();
                    let mut ps = vec![1usize; plen];
                    for k in (0..plen.saturating_sub(1)).rev() {
                        ps[k] = ps[k + 1] * cpd.parent_states[k + 1];
                    }
                    ps
                };
                for k in 0..cpd.parents.len() {
                    parent_states_vec[k] = tmp / parent_stride[k];
                    tmp %= parent_stride[k];
                }
            }

            for state in 0..cpd.num_states {
                // Build multi-index in factor (parents..., node)
                let mut flat = 0usize;
                for (k, &ps) in parent_states_vec.iter().enumerate() {
                    flat += ps * strides[k];
                }
                flat += state * strides[vars.len() - 1];
                values[flat] = cpd.table[parent_flat][state];
            }
        }

        Ok(Factor {
            vars,
            cardinalities: cards,
            values,
        })
    }

    /// Variable Elimination inference.
    ///
    /// Returns normalised marginals for each query node.
    pub fn variable_elimination(
        &self,
        query_nodes: &[&str],
        evidence: &HashMap<String, usize>,
    ) -> Result<HashMap<String, Vec<f64>>, StatsError> {
        // Build one factor per node
        let mut factors: Vec<Factor> = self
            .nodes
            .iter()
            .map(|n| self.node_factor(n.as_str()))
            .collect::<Result<Vec<_>, _>>()?;

        // Apply evidence: restrict each factor to observed values
        for (obs_var, &obs_state) in evidence {
            factors = factors
                .into_iter()
                .map(|f| {
                    if f.vars.contains(obs_var) {
                        f.observe(obs_var, obs_state)
                            .unwrap_or_else(|| Factor::new(vec![], vec![]))
                    } else {
                        f
                    }
                })
                .collect();
        }

        // Determine elimination order: all non-query, non-evidence nodes
        let query_set: HashSet<&str> = query_nodes.iter().copied().collect();
        let evidence_set: HashSet<&str> = evidence.keys().map(|k| k.as_str()).collect();

        let topo = self.topological_order()?;
        let elim_vars: Vec<String> = topo
            .iter()
            .filter(|n| !query_set.contains(n.as_str()) && !evidence_set.contains(n.as_str()))
            .cloned()
            .collect();

        // Eliminate each variable by summing out
        for var in &elim_vars {
            let (relevant, mut rest): (Vec<Factor>, Vec<Factor>) =
                factors.into_iter().partition(|f| f.vars.contains(var));
            if relevant.is_empty() {
                factors = rest;
                continue;
            }
            // Multiply all relevant factors
            let product: Option<Factor> = relevant.into_iter().reduce(|a, b| a.multiply(&b));
            if let Some(prod) = product {
                // Sum out the variable
                if let Some(marginalised) = prod.marginalize(var) {
                    if !marginalised.vars.is_empty() || marginalised.values.len() == 1 {
                        rest.push(marginalised);
                    }
                } else {
                    rest.push(prod);
                }
            }
            factors = rest;
        }

        // For each query node, collect its marginal
        let mut result = HashMap::new();
        for &qn in query_nodes {
            let relevant: Vec<Factor> = factors
                .iter()
                .filter(|f| f.vars.contains(&qn.to_string()))
                .cloned()
                .collect();

            let n_states = *self.node_states.get(qn).ok_or_else(|| {
                StatsError::InvalidArgument(format!("Query node '{}' not found", qn))
            })?;

            if relevant.is_empty() {
                result.insert(qn.to_string(), vec![1.0 / n_states as f64; n_states]);
                continue;
            }

            let combined: Option<Factor> = relevant.into_iter().reduce(|a, b| a.multiply(&b));
            let combined = match combined {
                Some(f) => f,
                None => {
                    result.insert(qn.to_string(), vec![1.0 / n_states as f64; n_states]);
                    continue;
                }
            };

            // Marginalise all variables except qn
            let mut current = combined;
            let other_vars: Vec<String> = current
                .vars
                .iter()
                .filter(|v| v.as_str() != qn)
                .cloned()
                .collect();
            for ov in &other_vars {
                if let Some(m) = current.marginalize(ov.as_str()) {
                    current = m;
                }
            }

            // Extract values for qn
            let mut marginal = vec![0.0_f64; n_states];
            for s in 0..n_states {
                marginal[s] = current.value_at(qn, s).unwrap_or(0.0);
            }

            // Normalise
            let sum: f64 = marginal.iter().sum();
            if sum > 0.0 {
                for v in &mut marginal {
                    *v /= sum;
                }
            } else {
                let inv = 1.0 / n_states as f64;
                for v in &mut marginal {
                    *v = inv;
                }
            }
            result.insert(qn.to_string(), marginal);
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Ancestral Sampling
    // -----------------------------------------------------------------------

    /// Draw `n` joint samples from the network using ancestral sampling.
    ///
    /// Uses a simple linear congruential generator for reproducibility.
    pub fn sample(
        &self,
        n: usize,
        rng_seed: u64,
    ) -> Result<Vec<HashMap<String, usize>>, StatsError> {
        let order = self.topological_order()?;

        // Simple LCG RNG (Numerical Recipes parameters)
        let mut state = rng_seed;
        let lcg_next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Use upper 32 bits for better quality; shift by 32 to get full [0, 1) range
            let upper = (*s >> 32) as f64;
            upper / ((1u64 << 32) as f64)
        };

        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let mut assignment: HashMap<String, usize> = HashMap::new();
            for node in &order {
                let cpd = self.cpds.get(node.as_str()).ok_or_else(|| {
                    StatsError::ComputationError(format!("No CPT for node '{}'", node))
                })?;

                let parent_config: Vec<usize> = cpd
                    .parents
                    .iter()
                    .map(|p| assignment.get(p.as_str()).copied().unwrap_or(0))
                    .collect();

                let row_idx = cpd.config_index(&parent_config)?;
                let row = &cpd.table[row_idx];

                // Sample from categorical using cumulative sum
                let u = lcg_next(&mut state);
                let mut cumsum = 0.0_f64;
                let mut chosen = row.len() - 1;
                for (k, &p) in row.iter().enumerate() {
                    cumsum += p;
                    if u <= cumsum {
                        chosen = k;
                        break;
                    }
                }
                assignment.insert(node.clone(), chosen);
            }
            samples.push(assignment);
        }
        Ok(samples)
    }

    // -----------------------------------------------------------------------
    // Markov Blanket
    // -----------------------------------------------------------------------

    /// Markov blanket of `node` = parents ∪ children ∪ co-parents (other parents of children).
    pub fn markov_blanket(&self, node: &str) -> Vec<String> {
        let mut mb = HashSet::new();
        let parents = self.parents(node);
        let children = self.children(node);

        for p in &parents {
            mb.insert(p.clone());
        }
        for c in &children {
            mb.insert(c.clone());
            for cp in self.parents(c) {
                if cp != node {
                    mb.insert(cp);
                }
            }
        }
        mb.remove(node);
        let mut result: Vec<String> = mb.into_iter().collect();
        result.sort();
        result
    }

    // -----------------------------------------------------------------------
    // d-separation
    // -----------------------------------------------------------------------

    /// Test whether `node1` and `node2` are d-separated given `observed`.
    ///
    /// Uses the Bayes Ball algorithm (Shachter 1998).
    pub fn d_separated(&self, node1: &str, node2: &str, observed: &[&str]) -> bool {
        let observed_set: HashSet<&str> = observed.iter().copied().collect();
        // Find all ancestors of observed nodes (for collider activation)
        let ancestors = self.ancestors_of(observed);

        // Bayes Ball algorithm (Shachter 1998):
        //   Ball going UP (came from child):
        //     node NOT observed → pass to parents (up) and children (down)
        //     node observed     → blocked
        //   Ball going DOWN (came from parent):
        //     node NOT observed → pass to children (down) only
        //     node observed or ancestor-of-observed → collider activation: pass to parents (up)
        let mut visited: HashSet<(String, bool)> = HashSet::new();
        let mut queue: VecDeque<(String, bool)> = VecDeque::new();
        queue.push_back((node1.to_string(), true));
        queue.push_back((node1.to_string(), false));

        while let Some((node, via_child)) = queue.pop_front() {
            if visited.contains(&(node.clone(), via_child)) {
                continue;
            }
            visited.insert((node.clone(), via_child));

            if node != node1 && node == node2 {
                return false; // active path found → not d-separated
            }

            let is_observed = observed_set.contains(node.as_str());

            if via_child {
                // Ball came from a child (going up)
                if !is_observed {
                    // Non-observed: pass through to parents and fork to children
                    for p in self.parents(node.as_str()) {
                        queue.push_back((p, true));
                    }
                    for c in self.children(node.as_str()) {
                        queue.push_back((c, false));
                    }
                }
                // If observed: blocked (do nothing)
            } else {
                // Ball came from a parent (going down)
                if !is_observed {
                    // Non-observed: continue downstream only
                    for c in self.children(node.as_str()) {
                        queue.push_back((c, false));
                    }
                }
                // Collider activation: if this node is observed or is an ancestor
                // of an observed node, bounce ball up to parents
                if is_observed || ancestors.contains(&node) {
                    for p in self.parents(node.as_str()) {
                        queue.push_back((p, true));
                    }
                }
            }
        }

        // node2 was never reached via any direction
        !visited.contains(&(node2.to_string(), true))
            && !visited.contains(&(node2.to_string(), false))
    }

    /// Compute the set of all ancestors of the given nodes (inclusive).
    fn ancestors_of(&self, nodes: &[&str]) -> HashSet<String> {
        let mut ancestors = HashSet::new();
        let mut queue: VecDeque<String> = nodes.iter().map(|n| n.to_string()).collect();
        while let Some(node) = queue.pop_front() {
            if ancestors.insert(node.clone()) {
                for p in self.parents(&node) {
                    queue.push_back(p);
                }
            }
        }
        ancestors
    }

    /// Expose edges (parent, child) for testing.
    pub fn edges(&self) -> &[(String, String)] {
        &self.edges
    }

    /// Expose nodes for testing.
    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build the classic Wet Grass network
    //   Rain → WetGrass ← Sprinkler
    fn wet_grass_network() -> BayesianNetwork {
        let mut bn = BayesianNetwork::new();
        bn.add_node("Rain", 2).unwrap();
        bn.add_node("Sprinkler", 2).unwrap();
        bn.add_node("WetGrass", 2).unwrap();
        bn.add_edge("Rain", "WetGrass").unwrap();
        bn.add_edge("Sprinkler", "WetGrass").unwrap();

        // P(Rain)
        let mut rain_cpd = ConditionalProbability::new("Rain", 2);
        rain_cpd.set_row(&[], vec![0.8, 0.2]).unwrap();

        // P(Sprinkler)
        let mut spr_cpd = ConditionalProbability::new("Sprinkler", 2);
        spr_cpd.set_row(&[], vec![0.5, 0.5]).unwrap();

        // P(WetGrass | Rain, Sprinkler)
        let mut wg_cpd = ConditionalProbability::with_parents(
            "WetGrass",
            2,
            vec!["Rain".to_string(), "Sprinkler".to_string()],
            vec![2, 2],
        );
        wg_cpd.set_row(&[0, 0], vec![0.99, 0.01]).unwrap();
        wg_cpd.set_row(&[0, 1], vec![0.1, 0.9]).unwrap();
        wg_cpd.set_row(&[1, 0], vec![0.1, 0.9]).unwrap();
        wg_cpd.set_row(&[1, 1], vec![0.01, 0.99]).unwrap();

        bn.add_cpd(rain_cpd).unwrap();
        bn.add_cpd(spr_cpd).unwrap();
        bn.add_cpd(wg_cpd).unwrap();
        bn
    }

    #[test]
    fn test_add_nodes_and_edges() {
        let mut bn = BayesianNetwork::new();
        bn.add_node("A", 2).unwrap();
        bn.add_node("B", 3).unwrap();
        bn.add_edge("A", "B").unwrap();
        assert_eq!(bn.nodes().len(), 2);
        assert_eq!(bn.edges().len(), 1);
    }

    #[test]
    fn test_duplicate_node_error() {
        let mut bn = BayesianNetwork::new();
        bn.add_node("A", 2).unwrap();
        assert!(bn.add_node("A", 2).is_err());
    }

    #[test]
    fn test_self_loop_error() {
        let mut bn = BayesianNetwork::new();
        bn.add_node("A", 2).unwrap();
        assert!(bn.add_edge("A", "A").is_err());
    }

    #[test]
    fn test_cycle_detection() {
        let mut bn = BayesianNetwork::new();
        bn.add_node("A", 2).unwrap();
        bn.add_node("B", 2).unwrap();
        bn.add_node("C", 2).unwrap();
        bn.add_edge("A", "B").unwrap();
        bn.add_edge("B", "C").unwrap();
        assert!(bn.add_edge("C", "A").is_err());
    }

    #[test]
    fn test_is_dag() {
        let bn = wet_grass_network();
        assert!(bn.is_dag());
    }

    #[test]
    fn test_topological_order() {
        let bn = wet_grass_network();
        let order = bn.topological_order().unwrap();
        let pos = |name: &str| order.iter().position(|n| n == name).unwrap();
        assert!(pos("Rain") < pos("WetGrass"));
        assert!(pos("Sprinkler") < pos("WetGrass"));
    }

    #[test]
    fn test_cpd_invalid_row_sum() {
        let mut cpd = ConditionalProbability::new("A", 2);
        assert!(cpd.set_row(&[], vec![0.3, 0.3]).is_err());
    }

    #[test]
    fn test_cpd_get_probability() {
        let mut cpd = ConditionalProbability::new("A", 2);
        cpd.set_row(&[], vec![0.3, 0.7]).unwrap();
        assert!((cpd.get_probability(0, &[]).unwrap() - 0.3).abs() < 1e-9);
        assert!((cpd.get_probability(1, &[]).unwrap() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_variable_elimination_prior() {
        let bn = wet_grass_network();
        let evidence = HashMap::new();
        let result = bn.variable_elimination(&["Rain"], &evidence).unwrap();
        let rain_marginal = &result["Rain"];
        assert!(
            (rain_marginal[0] - 0.8).abs() < 1e-4,
            "Got {:?}",
            rain_marginal
        );
        assert!(
            (rain_marginal[1] - 0.2).abs() < 1e-4,
            "Got {:?}",
            rain_marginal
        );
    }

    #[test]
    fn test_variable_elimination_with_evidence() {
        let bn = wet_grass_network();
        let mut evidence = HashMap::new();
        evidence.insert("WetGrass".to_string(), 1usize);
        let result = bn.variable_elimination(&["Rain"], &evidence).unwrap();
        let rain_marginal = &result["Rain"];
        assert!(
            rain_marginal[1] > 0.2,
            "Expected P(Rain=1|WetGrass=1) > 0.2, got {:?}",
            rain_marginal
        );
    }

    #[test]
    fn test_ancestral_sampling_count() {
        let bn = wet_grass_network();
        let samples = bn.sample(100, 42).unwrap();
        assert_eq!(samples.len(), 100);
        for s in &samples {
            assert!(s.contains_key("Rain"));
            assert!(s.contains_key("Sprinkler"));
            assert!(s.contains_key("WetGrass"));
        }
    }

    #[test]
    fn test_ancestral_sampling_state_range() {
        let bn = wet_grass_network();
        let samples = bn.sample(200, 99).unwrap();
        for s in &samples {
            assert!(*s.get("Rain").unwrap() < 2);
            assert!(*s.get("Sprinkler").unwrap() < 2);
            assert!(*s.get("WetGrass").unwrap() < 2);
        }
    }

    #[test]
    fn test_ancestral_sampling_marginal() {
        let bn = wet_grass_network();
        let n = 5000;
        let samples = bn.sample(n, 7).unwrap();
        let rain_1 = samples
            .iter()
            .filter(|s| *s.get("Rain").unwrap() == 1)
            .count();
        let freq = rain_1 as f64 / n as f64;
        assert!(
            (freq - 0.2).abs() < 0.05,
            "P(Rain=1) frequency={:.3}, expected ~0.2",
            freq
        );
    }

    #[test]
    fn test_markov_blanket() {
        let mut bn = BayesianNetwork::new();
        for n in &["A", "B", "C", "D"] {
            bn.add_node(n, 2).unwrap();
        }
        bn.add_edge("A", "B").unwrap();
        bn.add_edge("C", "B").unwrap();
        bn.add_edge("C", "D").unwrap();
        let mb = bn.markov_blanket("B");
        assert!(mb.contains(&"A".to_string()));
        assert!(mb.contains(&"C".to_string()));
        let mb_c = bn.markov_blanket("C");
        assert!(mb_c.contains(&"B".to_string()));
        assert!(mb_c.contains(&"D".to_string()));
        assert!(mb_c.contains(&"A".to_string()));
    }

    #[test]
    fn test_d_separation_simple_chain() {
        // A → B → C; given B, A ⊥ C
        let mut bn = BayesianNetwork::new();
        for n in &["A", "B", "C"] {
            bn.add_node(n, 2).unwrap();
        }
        bn.add_edge("A", "B").unwrap();
        bn.add_edge("B", "C").unwrap();
        assert!(
            bn.d_separated("A", "C", &["B"]),
            "A and C should be d-separated given B"
        );
        assert!(
            !bn.d_separated("A", "C", &[]),
            "A and C should NOT be d-separated marginally"
        );
    }

    #[test]
    fn test_d_separation_v_structure() {
        // A → B ← C; A ⊥ C marginally (v-structure)
        let mut bn = BayesianNetwork::new();
        for n in &["A", "B", "C"] {
            bn.add_node(n, 2).unwrap();
        }
        bn.add_edge("A", "B").unwrap();
        bn.add_edge("C", "B").unwrap();
        assert!(
            bn.d_separated("A", "C", &[]),
            "A and C should be d-separated marginally (v-structure)"
        );
    }

    #[test]
    fn test_parents_children() {
        let bn = wet_grass_network();
        let rain_children = bn.children("Rain");
        assert!(rain_children.contains(&"WetGrass".to_string()));
        let wg_parents = bn.parents("WetGrass");
        assert!(wg_parents.contains(&"Rain".to_string()));
        assert!(wg_parents.contains(&"Sprinkler".to_string()));
    }

    #[test]
    fn test_node_factor_no_parents() {
        let bn = wet_grass_network();
        let factor = bn.node_factor("Rain").unwrap();
        assert_eq!(factor.vars, vec!["Rain".to_string()]);
        assert_eq!(factor.values.len(), 2);
        assert!((factor.values[0] - 0.8).abs() < 1e-9);
        assert!((factor.values[1] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_sprinkler_prior() {
        let bn = wet_grass_network();
        let evidence = HashMap::new();
        let result = bn.variable_elimination(&["Sprinkler"], &evidence).unwrap();
        let s = &result["Sprinkler"];
        assert!((s[0] - 0.5).abs() < 1e-4, "P(Spr=0) ~0.5, got {:?}", s);
    }

    #[test]
    fn test_variable_elimination_multiple_query() {
        let bn = wet_grass_network();
        let evidence = HashMap::new();
        let result = bn
            .variable_elimination(&["Rain", "Sprinkler"], &evidence)
            .unwrap();
        assert!(result.contains_key("Rain"));
        assert!(result.contains_key("Sprinkler"));
    }

    #[test]
    fn test_add_cpd_unknown_node() {
        let mut bn = BayesianNetwork::new();
        let cpd = ConditionalProbability::new("Unknown", 2);
        assert!(bn.add_cpd(cpd).is_err());
    }
}
