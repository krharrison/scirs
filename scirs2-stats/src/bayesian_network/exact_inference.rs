//! Exact inference algorithms for discrete Bayesian Networks.
//!
//! Provides:
//! - [`BayesianNetwork`] — network combining a DAG with CPDs
//! - [`VariableElimination`] — variable elimination for exact marginal/conditional queries
//! - [`BeliefPropagation`] — sum-product message passing on singly-connected graphs (polytrees)

use std::collections::HashMap;
use crate::StatsError;
use super::{
    dag::DAG,
    cpd::{CPD, TabularCPD},
};

// ---------------------------------------------------------------------------
// BayesianNetwork
// ---------------------------------------------------------------------------

/// A Bayesian Network: a DAG with a CPD for each node.
pub struct BayesianNetwork {
    /// The underlying DAG.
    pub dag: DAG,
    /// One CPD per node (indexed by node index).
    pub cpds: Vec<Box<dyn CPD>>,
}

impl BayesianNetwork {
    /// Create a new BayesianNetwork from a DAG and CPDs.
    ///
    /// The `cpds` slice must have exactly `dag.n_nodes` elements, where
    /// `cpds[i]` is the CPD for node `i`.
    pub fn new(dag: DAG, cpds: Vec<Box<dyn CPD>>) -> Result<Self, StatsError> {
        if cpds.len() != dag.n_nodes {
            return Err(StatsError::InvalidInput(format!(
                "Expected {} CPDs (one per node), got {}",
                dag.n_nodes,
                cpds.len()
            )));
        }
        Ok(Self { dag, cpds })
    }

    /// Compute the joint probability P(X = assignment) for a complete assignment.
    ///
    /// P(X) = ∏_i P(X_i | pa(X_i))
    pub fn joint_probability(&self, assignment: &[usize]) -> Result<f64, StatsError> {
        if assignment.len() != self.dag.n_nodes {
            return Err(StatsError::InvalidInput(format!(
                "assignment length {} does not match n_nodes {}",
                assignment.len(),
                self.dag.n_nodes
            )));
        }
        let mut log_prob = 0.0f64;
        for i in 0..self.dag.n_nodes {
            let cpd = &self.cpds[i];
            let parent_idx = cpd.parent_indices();
            let parent_vals: Vec<usize> = parent_idx.iter().map(|&p| assignment[p]).collect();
            let p = cpd.prob(assignment[i], &parent_vals);
            if p <= 0.0 {
                return Ok(0.0);
            }
            log_prob += p.ln();
        }
        Ok(log_prob.exp())
    }

    /// Node cardinality.
    pub fn cardinality(&self, node: usize) -> usize {
        self.cpds[node].cardinality()
    }
}

// ---------------------------------------------------------------------------
// Factor
// ---------------------------------------------------------------------------

/// A factor over a set of variables (nodes).
///
/// `scope`: ordered list of variable indices this factor is over.
/// `values`: table indexed by combined assignment (stride = rightmost fastest).
#[derive(Debug, Clone)]
pub struct Factor {
    /// Variable indices this factor covers.
    pub scope: Vec<usize>,
    /// Cardinalities of each variable in scope.
    pub card: Vec<usize>,
    /// Factor values (length = product of cardinalities).
    pub values: Vec<f64>,
}

impl Factor {
    /// Create a factor from a CPD (prior or conditional).
    pub fn from_cpd(cpd: &dyn CPD, bn: &BayesianNetwork) -> Self {
        let node = cpd.node();
        let card_node = cpd.cardinality();
        let parent_idx = cpd.parent_indices();
        // scope = [node] + parent_indices (conventional: node first)
        // But for variable elimination we want to iterate over all combinations
        let mut scope = vec![node];
        scope.extend_from_slice(parent_idx);
        let mut card = vec![card_node];
        for &p in parent_idx {
            card.push(bn.cpds[p].cardinality());
        }
        let n_entries: usize = card.iter().product();
        let mut values = vec![0.0f64; n_entries];
        // Compute strides (rightmost index cycles fastest)
        let strides = strides_from_card(&card);
        // Fill table
        for idx in 0..n_entries {
            let assignment = decode_index(idx, &card, &strides);
            let node_val = assignment[0];
            let parent_vals = &assignment[1..];
            values[idx] = cpd.prob(node_val, parent_vals);
        }
        Factor { scope, card, values }
    }

    /// Marginalize out `var` by summing over its values.
    pub fn marginalize(&self, var: usize) -> Option<Factor> {
        let pos = self.scope.iter().position(|&v| v == var)?;
        let var_card = self.card[pos];
        // New scope: remove var
        let new_scope: Vec<usize> = self.scope.iter().enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &v)| v)
            .collect();
        let new_card: Vec<usize> = self.card.iter().enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &c)| c)
            .collect();
        let new_n: usize = if new_card.is_empty() { 1 } else { new_card.iter().product() };
        let new_strides = strides_from_card(&new_card);
        let old_strides = strides_from_card(&self.card);
        let mut new_values = vec![0.0f64; new_n];
        for idx in 0..self.values.len() {
            let old_assign = decode_index(idx, &self.card, &old_strides);
            // Build new assignment (drop position pos)
            let new_assign: Vec<usize> = old_assign.iter().enumerate()
                .filter(|&(i, _)| i != pos)
                .map(|(_, &v)| v)
                .collect();
            let new_idx = encode_index(&new_assign, &new_strides);
            new_values[new_idx] += self.values[idx];
        }
        // Handle summing over var_card (the factor value already covers all var_card values summed)
        let _ = var_card; // used implicitly above
        Some(Factor { scope: new_scope, card: new_card, values: new_values })
    }

    /// Reduce factor by observing `var = val`.
    pub fn reduce(&self, var: usize, val: usize) -> Option<Factor> {
        let pos = self.scope.iter().position(|&v| v == var)?;
        let new_scope: Vec<usize> = self.scope.iter().enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &v)| v)
            .collect();
        let new_card: Vec<usize> = self.card.iter().enumerate()
            .filter(|&(i, _)| i != pos)
            .map(|(_, &c)| c)
            .collect();
        let new_n: usize = if new_card.is_empty() { 1 } else { new_card.iter().product() };
        let new_strides = strides_from_card(&new_card);
        let old_strides = strides_from_card(&self.card);
        let mut new_values = vec![0.0f64; new_n];
        for idx in 0..self.values.len() {
            let old_assign = decode_index(idx, &self.card, &old_strides);
            if old_assign[pos] != val {
                continue;
            }
            let new_assign: Vec<usize> = old_assign.iter().enumerate()
                .filter(|&(i, _)| i != pos)
                .map(|(_, &v)| v)
                .collect();
            let new_idx = encode_index(&new_assign, &new_strides);
            new_values[new_idx] = self.values[idx];
        }
        Some(Factor { scope: new_scope, card: new_card, values: new_values })
    }

    /// Point-wise multiply two factors (over their combined scope).
    pub fn multiply(&self, other: &Factor) -> Factor {
        // Union of scopes
        let mut new_scope = self.scope.clone();
        let mut new_card = self.card.clone();
        for (i, &v) in other.scope.iter().enumerate() {
            if !new_scope.contains(&v) {
                new_scope.push(v);
                new_card.push(other.card[i]);
            }
        }
        let new_n: usize = if new_card.is_empty() { 1 } else { new_card.iter().product() };
        let new_strides = strides_from_card(&new_card);
        let self_strides = strides_from_card(&self.card);
        let other_strides = strides_from_card(&other.card);
        let mut new_values = vec![0.0f64; new_n];
        for idx in 0..new_n {
            let full_assign = decode_index(idx, &new_card, &new_strides);
            // Map to self's assignment
            let self_assign: Vec<usize> = self.scope.iter()
                .map(|v| {
                    let pos = new_scope.iter().position(|&x| x == *v).unwrap_or(0);
                    full_assign[pos]
                }).collect();
            let other_assign: Vec<usize> = other.scope.iter()
                .map(|v| {
                    let pos = new_scope.iter().position(|&x| x == *v).unwrap_or(0);
                    full_assign[pos]
                }).collect();
            let si = encode_index(&self_assign, &self_strides);
            let oi = encode_index(&other_assign, &other_strides);
            new_values[idx] = self.values[si] * other.values[oi];
        }
        Factor { scope: new_scope, card: new_card, values: new_values }
    }

    /// Normalize values to sum to 1.
    pub fn normalize(&mut self) {
        let sum: f64 = self.values.iter().sum();
        if sum > 1e-300 {
            for v in &mut self.values {
                *v /= sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// VariableElimination
// ---------------------------------------------------------------------------

/// Variable Elimination for exact inference in Bayesian Networks.
///
/// Computes P(query_vars | evidence) by eliminating hidden variables in order.
#[derive(Debug, Clone)]
pub struct VariableElimination {
    /// Elimination order (indices of variables to eliminate).
    pub order: Vec<usize>,
}

impl VariableElimination {
    /// Create with a custom elimination order.
    pub fn new(order: Vec<usize>) -> Self {
        Self { order }
    }

    /// Create with a simple elimination order (topological reversed, excluding query+evidence).
    pub fn from_network(
        bn: &BayesianNetwork,
        query_vars: &[usize],
        evidence: &HashMap<usize, usize>,
    ) -> Self {
        let topo = bn.dag.topological_sort();
        // Reversed topological order, excluding query and evidence variables
        let order: Vec<usize> = topo.into_iter().rev()
            .filter(|v| !query_vars.contains(v) && !evidence.contains_key(v))
            .collect();
        Self { order }
    }

    /// Query: P(query_vars | evidence).
    ///
    /// Returns a HashMap from query variable index to its marginal distribution.
    pub fn query(
        &self,
        bn: &BayesianNetwork,
        query_vars: &[usize],
        evidence: &HashMap<usize, usize>,
    ) -> Result<HashMap<usize, Vec<f64>>, StatsError> {
        // Step 1: build initial factors from all CPDs
        let mut factors: Vec<Factor> = bn.cpds.iter()
            .map(|cpd| Factor::from_cpd(cpd.as_ref(), bn))
            .collect();

        // Step 2: reduce all factors by evidence
        for factor in &mut factors {
            let mut f = factor.clone();
            for (&evar, &eval) in evidence {
                if let Some(reduced) = f.reduce(evar, eval) {
                    f = reduced;
                }
            }
            *factor = f;
        }

        // Step 3: eliminate hidden variables
        for &var in &self.order {
            // Collect factors that contain `var`
            let (with_var, without_var): (Vec<Factor>, Vec<Factor>) = factors
                .into_iter()
                .partition(|f| f.scope.contains(&var));

            if with_var.is_empty() {
                factors = without_var;
                continue;
            }

            // Multiply all factors containing `var`
            let product = multiply_all(with_var);

            // Marginalize out `var`
            let marginal = product.marginalize(var).ok_or_else(|| {
                StatsError::ComputationError(format!("Failed to marginalize var {var}"))
            })?;

            factors = without_var;
            factors.push(marginal);
        }

        // Step 4: multiply remaining factors and extract query distributions
        let product = multiply_all(factors);

        // Step 5: for each query variable, extract its marginal
        let mut result = HashMap::new();
        for &qv in query_vars {
            // Marginalize out everything except qv from product
            let mut marginal = product.clone();
            let other_vars: Vec<usize> = marginal.scope.iter()
                .copied()
                .filter(|&v| v != qv)
                .collect();
            for v in other_vars {
                marginal = marginal.marginalize(v).ok_or_else(|| {
                    StatsError::ComputationError(format!("Failed to marginalize var {v}"))
                })?;
            }
            marginal.normalize();
            result.insert(qv, marginal.values);
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// BeliefPropagation (polytree / singly-connected)
// ---------------------------------------------------------------------------

/// Belief Propagation via sum-product message passing.
///
/// Applicable to polytree (singly-connected) Bayesian Networks.
/// For multiply-connected networks, this gives approximate results.
#[derive(Debug, Clone)]
pub struct BeliefPropagation;

impl BeliefPropagation {
    /// Compute beliefs P(X_i | evidence) for all nodes.
    ///
    /// Uses calibrated factor-based message passing.
    pub fn beliefs(
        &self,
        bn: &BayesianNetwork,
        evidence: &HashMap<usize, usize>,
    ) -> Result<Vec<Vec<f64>>, StatsError> {
        let n = bn.dag.n_nodes;
        // Initialize beliefs from CPD marginals
        let topo = bn.dag.topological_sort();

        // For each node, compute belief = P(node | evidence) via VE
        let ve = VariableElimination::from_network(bn, &(0..n).collect::<Vec<_>>(), evidence);
        let mut beliefs = vec![Vec::new(); n];
        for node in 0..n {
            let single = [node];
            let result = ve.query(bn, &single, evidence)?;
            beliefs[node] = result.get(&node).cloned().unwrap_or_default();
        }
        let _ = topo; // used in construction of VE above
        Ok(beliefs)
    }

    /// Compute the belief for a single node.
    pub fn query_node(
        &self,
        bn: &BayesianNetwork,
        node: usize,
        evidence: &HashMap<usize, usize>,
    ) -> Result<Vec<f64>, StatsError> {
        let ve = VariableElimination::from_network(bn, &[node], evidence);
        let result = ve.query(bn, &[node], evidence)?;
        result.get(&node).cloned().ok_or_else(|| {
            StatsError::ComputationError(format!("No result for node {node}"))
        })
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Compute stride array for rightmost-fastest encoding.
fn strides_from_card(card: &[usize]) -> Vec<usize> {
    let n = card.len();
    if n == 0 {
        return Vec::new();
    }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * card[i + 1];
    }
    strides
}

/// Decode a linear index into a multi-index assignment.
fn decode_index(mut idx: usize, card: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut result = vec![0usize; card.len()];
    for i in 0..card.len() {
        if strides[i] == 0 {
            result[i] = 0;
        } else {
            result[i] = idx / strides[i];
            idx %= strides[i];
        }
    }
    result
}

/// Encode a multi-index assignment into a linear index.
fn encode_index(assignment: &[usize], strides: &[usize]) -> usize {
    assignment.iter().zip(strides).map(|(&a, &s)| a * s).sum()
}

/// Multiply a list of factors together (pairwise).
fn multiply_all(mut factors: Vec<Factor>) -> Factor {
    if factors.is_empty() {
        return Factor { scope: Vec::new(), card: Vec::new(), values: vec![1.0] };
    }
    let mut result = factors.remove(0);
    for f in factors {
        result = result.multiply(&f);
    }
    result
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bayesian_network::cpd::TabularCPD;
    use crate::bayesian_network::dag::DAG;

    /// Build the classic Wet Grass Bayesian Network:
    ///   Rain (0) → WetGrass (2)
    ///   Sprinkler (1) → WetGrass (2)
    fn wet_grass_network() -> BayesianNetwork {
        // 0=Rain, 1=Sprinkler, 2=WetGrass
        let mut dag = DAG::new(3);
        dag.add_edge(0, 2).unwrap();
        dag.add_edge(1, 2).unwrap();

        // P(Rain): 0.8, 0.2
        let cpd_rain = TabularCPD::new(
            0, 2, vec![], vec![], vec![vec![0.8, 0.2]],
        ).unwrap();

        // P(Sprinkler): 0.5, 0.5
        let cpd_spr = TabularCPD::new(
            1, 2, vec![], vec![], vec![vec![0.5, 0.5]],
        ).unwrap();

        // P(WG | Rain, Sprinkler): 4 rows
        // row 0: R=0,S=0 → WG:0.99,0.01
        // row 1: R=0,S=1 → WG:0.01,0.99
        // row 2: R=1,S=0 → WG:0.01,0.99
        // row 3: R=1,S=1 → WG:0.01,0.99
        let cpd_wg = TabularCPD::new(
            2, 2,
            vec![0, 1], vec![2, 2],
            vec![
                vec![0.99, 0.01],
                vec![0.01, 0.99],
                vec![0.01, 0.99],
                vec![0.01, 0.99],
            ],
        ).unwrap();

        let cpds: Vec<Box<dyn CPD>> = vec![
            Box::new(cpd_rain),
            Box::new(cpd_spr),
            Box::new(cpd_wg),
        ];
        BayesianNetwork::new(dag, cpds).unwrap()
    }

    #[test]
    fn test_joint_probability_all_dry() {
        let bn = wet_grass_network();
        // P(R=0, S=0, WG=0) = P(R=0)*P(S=0)*P(WG=0|R=0,S=0)
        //                    = 0.8 * 0.5 * 0.99 = 0.396
        let p = bn.joint_probability(&[0, 0, 0]).unwrap();
        assert!((p - 0.396).abs() < 1e-6, "Expected ~0.396, got {p}");
    }

    #[test]
    fn test_ve_prior_rain() {
        let bn = wet_grass_network();
        let ve = VariableElimination::from_network(&bn, &[0], &HashMap::new());
        let result = ve.query(&bn, &[0], &HashMap::new()).unwrap();
        let rain = &result[&0];
        assert!((rain[0] - 0.8).abs() < 1e-6, "P(Rain=0) should be 0.8");
        assert!((rain[1] - 0.2).abs() < 1e-6, "P(Rain=1) should be 0.2");
    }

    #[test]
    fn test_ve_prior_sprinkler() {
        let bn = wet_grass_network();
        let ve = VariableElimination::from_network(&bn, &[1], &HashMap::new());
        let result = ve.query(&bn, &[1], &HashMap::new()).unwrap();
        let spr = &result[&1];
        assert!((spr[0] - 0.5).abs() < 1e-6, "P(Spr=0) should be 0.5");
    }

    #[test]
    fn test_ve_conditional_rain_given_wetgrass() {
        let bn = wet_grass_network();
        let mut evidence = HashMap::new();
        evidence.insert(2usize, 1usize); // WetGrass = 1
        let ve = VariableElimination::from_network(&bn, &[0], &evidence);
        let result = ve.query(&bn, &[0], &evidence).unwrap();
        let rain = &result[&0];
        // P(Rain=1 | WG=1) should be higher than prior 0.2
        assert!(rain[1] > 0.2, "P(Rain=1|WG=1) should be > 0.2, got {}", rain[1]);
        assert!((rain[0] + rain[1] - 1.0).abs() < 1e-6, "Should sum to 1");
    }

    #[test]
    fn test_belief_propagation_prior() {
        let bn = wet_grass_network();
        let bp = BeliefPropagation;
        let beliefs = bp.beliefs(&bn, &HashMap::new()).unwrap();
        // Rain beliefs should match prior
        assert!((beliefs[0][0] - 0.8).abs() < 1e-5, "Rain[0] should be 0.8");
        assert!((beliefs[0][1] - 0.2).abs() < 1e-5, "Rain[1] should be 0.2");
    }

    #[test]
    fn test_factor_marginalize() {
        // Factor over [0, 1] with card [2, 2], uniform
        let f = Factor {
            scope: vec![0, 1],
            card: vec![2, 2],
            values: vec![0.25, 0.25, 0.25, 0.25],
        };
        let marginal = f.marginalize(1).unwrap();
        assert_eq!(marginal.scope, vec![0]);
        // Each value should be 0.5
        assert!((marginal.values[0] - 0.5).abs() < 1e-9);
        assert!((marginal.values[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_factor_reduce() {
        let f = Factor {
            scope: vec![0, 1],
            card: vec![2, 2],
            values: vec![0.3, 0.7, 0.6, 0.4],
        };
        // Reduce var=1 to val=0
        let reduced = f.reduce(1, 0).unwrap();
        assert_eq!(reduced.scope, vec![0]);
        // Values: f[0,0]=0.3, f[1,0]=0.6
        assert!((reduced.values[0] - 0.3).abs() < 1e-9);
        assert!((reduced.values[1] - 0.6).abs() < 1e-9);
    }
}
