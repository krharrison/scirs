//! Pairwise Markov Random Field (undirected graphical model).
//!
//! Provides:
//! - Construction with unary and pairwise log-potentials
//! - Gibbs sampling
//! - Loopy Belief Propagation (sum-product and max-product)
//! - Ising model constructor

use std::collections::HashMap;

use scirs2_core::ndarray::Array2;

use crate::error::StatsError;

// ---------------------------------------------------------------------------
// Markov Random Field
// ---------------------------------------------------------------------------

/// Pairwise Markov Random Field over discrete variables.
///
/// The unnormalised probability is:
/// ```text
/// P(x) ∝ exp(Σ_i unary[i][x_i]  +  Σ_{(i,j)∈E} pairwise[(i,j)][x_i][x_j])
/// ```
/// where all potentials are stored in **log-space**.
#[derive(Debug, Clone)]
pub struct MarkovRandomField {
    /// Number of variable nodes.
    pub n_nodes: usize,
    /// Number of discrete states per node (same for all nodes).
    pub n_states: usize,
    /// Unary log-potentials: `unary_potentials[node][state]`.
    pub unary_potentials: Vec<Vec<f64>>,
    /// Pairwise log-potentials: key `(i, j)` with i < j,
    /// value is `Array2` of shape `(n_states, n_states)`.
    pub pairwise_potentials: HashMap<(usize, usize), Array2<f64>>,
}

impl MarkovRandomField {
    /// Create an MRF with zero log-potentials.
    pub fn new(n_nodes: usize, n_states: usize) -> Self {
        MarkovRandomField {
            n_nodes,
            n_states,
            unary_potentials: vec![vec![0.0_f64; n_states]; n_nodes],
            pairwise_potentials: HashMap::new(),
        }
    }

    /// Set the unary log-potentials for `node`.
    pub fn set_unary(&mut self, node: usize, log_potentials: Vec<f64>) -> Result<(), StatsError> {
        if node >= self.n_nodes {
            return Err(StatsError::InvalidArgument(format!(
                "Node index {} out of range (n_nodes={})",
                node, self.n_nodes
            )));
        }
        if log_potentials.len() != self.n_states {
            return Err(StatsError::InvalidArgument(format!(
                "Expected {} log-potentials, got {}",
                self.n_states,
                log_potentials.len()
            )));
        }
        self.unary_potentials[node] = log_potentials;
        Ok(())
    }

    /// Add (or replace) a pairwise log-potential between nodes `i` and `j`.
    ///
    /// `log_potentials` must have shape `(n_states, n_states)`.
    pub fn add_edge(
        &mut self,
        node_i: usize,
        node_j: usize,
        log_potentials: Array2<f64>,
    ) -> Result<(), StatsError> {
        if node_i >= self.n_nodes || node_j >= self.n_nodes {
            return Err(StatsError::InvalidArgument(format!(
                "Node indices ({}, {}) out of range (n_nodes={})",
                node_i, node_j, self.n_nodes
            )));
        }
        if node_i == node_j {
            return Err(StatsError::InvalidArgument(
                "Self-loops are not allowed".to_string(),
            ));
        }
        if log_potentials.shape() != [self.n_states, self.n_states] {
            return Err(StatsError::InvalidArgument(format!(
                "Pairwise potential must be ({0},{0}), got {1:?}",
                self.n_states,
                log_potentials.shape()
            )));
        }
        let key = if node_i < node_j {
            (node_i, node_j)
        } else {
            (node_j, node_i)
        };
        // If stored as (i,j) with i<j, transpose when needed
        let stored = if node_i < node_j {
            log_potentials
        } else {
            log_potentials.t().to_owned()
        };
        self.pairwise_potentials.insert(key, stored);
        Ok(())
    }

    /// Returns the neighbours of `node` (nodes connected by an edge).
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        let mut nb: Vec<usize> = self
            .pairwise_potentials
            .keys()
            .filter_map(|&(i, j)| {
                if i == node {
                    Some(j)
                } else if j == node {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        nb.sort_unstable();
        nb
    }

    /// Get the pairwise log-potential `log ψ(x_i, x_j)` for nodes `i` and `j`.
    fn pairwise_log_potential(&self, node_i: usize, node_j: usize, si: usize, sj: usize) -> f64 {
        let key = if node_i < node_j {
            (node_i, node_j)
        } else {
            (node_j, node_i)
        };
        if let Some(table) = self.pairwise_potentials.get(&key) {
            if node_i < node_j {
                table[[si, sj]]
            } else {
                table[[sj, si]]
            }
        } else {
            0.0
        }
    }

    /// Compute the unnormalised energy E(x) = −log P̃(x) for an assignment.
    pub fn energy(&self, assignment: &[usize]) -> Result<f64, StatsError> {
        if assignment.len() != self.n_nodes {
            return Err(StatsError::InvalidArgument(format!(
                "Assignment length {} != n_nodes {}",
                assignment.len(),
                self.n_nodes
            )));
        }
        let mut energy = 0.0_f64;
        for (node, &state) in assignment.iter().enumerate() {
            if state >= self.n_states {
                return Err(StatsError::InvalidArgument(format!(
                    "State {} out of range for node {}",
                    state, node
                )));
            }
            energy -= self.unary_potentials[node][state];
        }
        for (&(i, j), table) in &self.pairwise_potentials {
            energy -= table[[assignment[i], assignment[j]]];
        }
        Ok(energy)
    }

    // -----------------------------------------------------------------------
    // Gibbs Sampling
    // -----------------------------------------------------------------------

    /// Gibbs sampling from the MRF.
    ///
    /// Returns a `Vec` of state assignments (one `Vec<usize>` per retained sample).
    pub fn gibbs_sample(
        &self,
        n_samples: usize,
        burn_in: usize,
        thin: usize,
        rng_seed: u64,
    ) -> Result<Vec<Vec<usize>>, StatsError> {
        if self.n_nodes == 0 {
            return Err(StatsError::InvalidArgument(
                "MRF has no nodes".to_string(),
            ));
        }
        let thin = thin.max(1);

        // LCG RNG
        let mut rng_state = rng_seed;
        let lcg_next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 33) as f64) / (u32::MAX as f64)
        };

        // Initialise uniformly
        let mut current: Vec<usize> = (0..self.n_nodes)
            .map(|i| {
                let r = lcg_next(&mut rng_state);
                (r * self.n_states as f64) as usize % self.n_states
            })
            .collect();

        let total_iters = burn_in + n_samples * thin;
        let mut samples = Vec::with_capacity(n_samples);
        let mut collected = 0usize;

        for iter in 0..total_iters {
            // Single Gibbs sweep: update each node in sequence
            for node in 0..self.n_nodes {
                // Compute conditional log-probabilities for each state
                let mut log_probs: Vec<f64> = (0..self.n_states)
                    .map(|s| self.unary_potentials[node][s])
                    .collect();

                for nb in self.neighbors(node) {
                    for s in 0..self.n_states {
                        log_probs[s] +=
                            self.pairwise_log_potential(node, nb, s, current[nb]);
                    }
                }

                // Numerically stable softmax → categorical sample
                let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
                let z: f64 = probs.iter().sum();

                let u = lcg_next(&mut rng_state) * z;
                let mut cumsum = 0.0_f64;
                let mut chosen = self.n_states - 1;
                for (k, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if u <= cumsum {
                        chosen = k;
                        break;
                    }
                }
                current[node] = chosen;
            }

            // Collect sample after burn-in with thinning
            if iter >= burn_in {
                let offset = iter - burn_in;
                if offset % thin == thin - 1 {
                    samples.push(current.clone());
                    collected += 1;
                    if collected >= n_samples {
                        break;
                    }
                }
            }
        }

        Ok(samples)
    }

    // -----------------------------------------------------------------------
    // Loopy Belief Propagation (sum-product)
    // -----------------------------------------------------------------------

    /// Loopy Belief Propagation using sum-product.
    ///
    /// Returns approximate marginals `[node][state]`.
    pub fn belief_propagation(
        &self,
        max_iter: usize,
        tol: f64,
        damping: f64,
    ) -> Result<Vec<Vec<f64>>, StatsError> {
        if damping < 0.0 || damping >= 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "Damping must be in [0, 1), got {}",
                damping
            )));
        }

        let n = self.n_nodes;
        let k = self.n_states;

        // messages[i][j][s] = message from node i to node j for state s
        // Indexed as: edge_key (i,j) with i<j → msg_ij and msg_ji
        // We'll use a flat structure keyed by (src, dst)
        let mut messages: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
        // Initialise all messages to uniform
        for (&(i, j), _) in &self.pairwise_potentials {
            messages.insert((i, j), vec![1.0 / k as f64; k]);
            messages.insert((j, i), vec![1.0 / k as f64; k]);
        }

        let edges: Vec<(usize, usize)> = self.pairwise_potentials.keys().copied().collect();

        for _iter in 0..max_iter {
            let mut max_delta = 0.0_f64;

            // Update messages for each directed edge
            let mut new_messages: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

            for &(ei, ej) in &edges {
                // Message ei → ej  and  ej → ei
                for (src, dst) in &[(ei, ej), (ej, ei)] {
                    let src = *src;
                    let dst = *dst;
                    let mut new_msg = vec![0.0_f64; k];

                    for sj in 0..k {
                        let mut sum_over_si = 0.0_f64;
                        for si in 0..k {
                            // Factor = exp(pairwise(src, dst, si, sj))
                            let pairwise_val =
                                self.pairwise_log_potential(src, dst, si, sj).exp();

                            // Unary factor for src
                            let unary_val = self.unary_potentials[src][si].exp();

                            // Product of incoming messages to src (excluding from dst)
                            let mut prod_incoming = 1.0_f64;
                            for nb in self.neighbors(src) {
                                if nb != dst {
                                    let msg = messages
                                        .get(&(nb, src))
                                        .and_then(|m| m.get(si))
                                        .copied()
                                        .unwrap_or(1.0 / k as f64);
                                    prod_incoming *= msg;
                                }
                            }
                            sum_over_si += pairwise_val * unary_val * prod_incoming;
                        }
                        new_msg[sj] = sum_over_si;
                    }

                    // Normalise
                    let z: f64 = new_msg.iter().sum();
                    if z > 0.0 {
                        for v in &mut new_msg {
                            *v /= z;
                        }
                    } else {
                        for v in &mut new_msg {
                            *v = 1.0 / k as f64;
                        }
                    }

                    // Apply damping
                    if damping > 0.0 {
                        let old = messages
                            .get(&(src, dst))
                            .cloned()
                            .unwrap_or_else(|| vec![1.0 / k as f64; k]);
                        for (new_v, old_v) in new_msg.iter_mut().zip(&old) {
                            *new_v = (1.0 - damping) * (*new_v) + damping * old_v;
                        }
                        // Re-normalise after damping
                        let z2: f64 = new_msg.iter().sum();
                        if z2 > 0.0 {
                            for v in &mut new_msg {
                                *v /= z2;
                            }
                        }
                    }

                    // Track convergence
                    if let Some(old) = messages.get(&(src, dst)) {
                        let delta: f64 = new_msg
                            .iter()
                            .zip(old.iter())
                            .map(|(a, b)| (a - b).abs())
                            .sum();
                        if delta > max_delta {
                            max_delta = delta;
                        }
                    }

                    new_messages.insert((src, dst), new_msg);
                }
            }

            // Merge new messages
            for (key, msg) in new_messages {
                messages.insert(key, msg);
            }

            if max_delta < tol {
                break;
            }
        }

        // Compute beliefs
        let mut beliefs: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..k)
                    .map(|s| self.unary_potentials[i][s].exp())
                    .collect()
            })
            .collect();

        for i in 0..n {
            for nb in self.neighbors(i) {
                if let Some(msg) = messages.get(&(nb, i)) {
                    for s in 0..k {
                        beliefs[i][s] *= msg[s];
                    }
                }
            }
            // Normalise
            let z: f64 = beliefs[i].iter().sum();
            if z > 0.0 {
                for v in &mut beliefs[i] {
                    *v /= z;
                }
            } else {
                for v in &mut beliefs[i] {
                    *v = 1.0 / k as f64;
                }
            }
        }

        Ok(beliefs)
    }

    // -----------------------------------------------------------------------
    // Max-product (MAP)
    // -----------------------------------------------------------------------

    /// Max-product belief propagation for MAP inference (approximate).
    ///
    /// Returns the MAP state for each node.
    pub fn max_product(
        &self,
        max_iter: usize,
        tol: f64,
    ) -> Result<Vec<usize>, StatsError> {
        let n = self.n_nodes;
        let k = self.n_states;

        // Initialise messages (in log-space to avoid underflow)
        let mut messages: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
        let edges: Vec<(usize, usize)> = self.pairwise_potentials.keys().copied().collect();

        for &(i, j) in &edges {
            messages.insert((i, j), vec![0.0_f64; k]); // log uniform = 0 (normalised out)
            messages.insert((j, i), vec![0.0_f64; k]);
        }

        for _iter in 0..max_iter {
            let mut max_delta = 0.0_f64;
            let mut new_messages: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

            for &(ei, ej) in &edges {
                for (src, dst) in &[(ei, ej), (ej, ei)] {
                    let src = *src;
                    let dst = *dst;
                    let mut new_msg = vec![f64::NEG_INFINITY; k];

                    for sj in 0..k {
                        let mut max_over_si = f64::NEG_INFINITY;
                        for si in 0..k {
                            let pairwise_val =
                                self.pairwise_log_potential(src, dst, si, sj);
                            let unary_val = self.unary_potentials[src][si];

                            let mut sum_incoming_log = 0.0_f64;
                            for nb in self.neighbors(src) {
                                if nb != dst {
                                    let msg_val = messages
                                        .get(&(nb, src))
                                        .and_then(|m| m.get(si))
                                        .copied()
                                        .unwrap_or(0.0);
                                    sum_incoming_log += msg_val;
                                }
                            }
                            let val = pairwise_val + unary_val + sum_incoming_log;
                            if val > max_over_si {
                                max_over_si = val;
                            }
                        }
                        new_msg[sj] = max_over_si;
                    }

                    // Subtract max for numerical stability
                    let max_val = new_msg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    if max_val.is_finite() {
                        for v in &mut new_msg {
                            *v -= max_val;
                        }
                    } else {
                        new_msg = vec![0.0; k];
                    }

                    // Convergence check
                    if let Some(old) = messages.get(&(src, dst)) {
                        let delta: f64 = new_msg
                            .iter()
                            .zip(old.iter())
                            .map(|(a, b)| (a - b).abs())
                            .sum();
                        if delta > max_delta {
                            max_delta = delta;
                        }
                    }

                    new_messages.insert((src, dst), new_msg);
                }
            }

            for (key, msg) in new_messages {
                messages.insert(key, msg);
            }

            if max_delta < tol {
                break;
            }
        }

        // Decode MAP states
        let map_states: Vec<usize> = (0..n)
            .map(|i| {
                let mut scores: Vec<f64> = (0..k)
                    .map(|s| self.unary_potentials[i][s])
                    .collect();
                for nb in self.neighbors(i) {
                    if let Some(msg) = messages.get(&(nb, i)) {
                        for s in 0..k {
                            scores[s] += msg[s];
                        }
                    }
                }
                scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        Ok(map_states)
    }

    // -----------------------------------------------------------------------
    // Ising Model constructor
    // -----------------------------------------------------------------------

    /// Build an Ising model on an `n_rows × n_cols` grid.
    ///
    /// - `h`: external field (unary) – positive biases toward state 1.
    /// - `j`: coupling (pairwise) – positive favours alignment.
    ///
    /// States: 0 (spin −1) and 1 (spin +1).
    ///
    /// Mapping to standard Ising convention:
    /// - unary potential for state 0 (σ=−1): `+h` (external field +h when σ=−1 is unusual; see convention below)
    /// - unary potential for state 1 (σ=+1): `−h` effectively `h·σ`
    ///
    /// Using convention: unary[0] = -h (spin -1 costs h), unary[1] = h (spin +1 gains h).
    pub fn ising_model(n_rows: usize, n_cols: usize, h: f64, j: f64) -> Self {
        let n_nodes = n_rows * n_cols;
        let mut mrf = MarkovRandomField::new(n_nodes, 2);

        // Set unary potentials: state 0 → spin -1, state 1 → spin +1
        for node in 0..n_nodes {
            // log P ∝ h * sigma, sigma ∈ {-1, +1}
            // state=0 → sigma=-1 → h * (-1) = -h
            // state=1 → sigma=+1 → h * (+1) = +h
            mrf.unary_potentials[node] = vec![-h, h];
        }

        // Pairwise potentials: log ψ(xi, xj) = j * σi * σj
        // (0,0): j * (-1)*(-1) = j
        // (0,1): j * (-1)*(+1) = -j
        // (1,0): j * (+1)*(-1) = -j
        // (1,1): j * (+1)*(+1) = j
        let pairwise = Array2::from_shape_vec(
            (2, 2),
            vec![j, -j, -j, j],
        )
        .expect("shape (2,2) with 4 elements is always valid");

        for row in 0..n_rows {
            for col in 0..n_cols {
                let node = row * n_cols + col;
                // Right neighbour
                if col + 1 < n_cols {
                    let right = row * n_cols + (col + 1);
                    mrf.pairwise_potentials.insert((node, right), pairwise.clone());
                }
                // Down neighbour
                if row + 1 < n_rows {
                    let down = (row + 1) * n_cols + col;
                    mrf.pairwise_potentials.insert((node, down), pairwise.clone());
                }
            }
        }

        mrf
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn ferromagnetic_ising(n: usize) -> MarkovRandomField {
        // 1D Ising chain: strong ferromagnetic coupling
        MarkovRandomField::ising_model(1, n, 0.0, 2.0)
    }

    #[test]
    fn test_mrf_construction() {
        let mrf = MarkovRandomField::new(4, 3);
        assert_eq!(mrf.n_nodes, 4);
        assert_eq!(mrf.n_states, 3);
    }

    #[test]
    fn test_set_unary() {
        let mut mrf = MarkovRandomField::new(2, 2);
        mrf.set_unary(0, vec![1.0, -1.0]).unwrap();
        assert_eq!(mrf.unary_potentials[0], vec![1.0, -1.0]);
    }

    #[test]
    fn test_set_unary_wrong_size() {
        let mut mrf = MarkovRandomField::new(2, 2);
        assert!(mrf.set_unary(0, vec![1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_add_edge() {
        let mut mrf = MarkovRandomField::new(3, 2);
        let pot = array![[0.5, -0.5], [-0.5, 0.5]];
        mrf.add_edge(0, 1, pot).unwrap();
        assert!(mrf.pairwise_potentials.contains_key(&(0, 1)));
    }

    #[test]
    fn test_add_edge_wrong_shape() {
        let mut mrf = MarkovRandomField::new(3, 2);
        let pot = array![[0.5, -0.5, 0.0], [-0.5, 0.5, 0.0]];
        assert!(mrf.add_edge(0, 1, pot).is_err());
    }

    #[test]
    fn test_energy_computation() {
        let mut mrf = MarkovRandomField::new(2, 2);
        // unary: node 0 prefers state 0, node 1 prefers state 1
        mrf.set_unary(0, vec![1.0, -1.0]).unwrap();
        mrf.set_unary(1, vec![-1.0, 1.0]).unwrap();
        // Energy = -(unary[0][0] + unary[1][1]) = -(1.0 + 1.0) = -2.0 (no pairwise)
        let e = mrf.energy(&[0, 1]).unwrap();
        assert!((e - (-2.0)).abs() < 1e-9, "energy={}", e);
    }

    #[test]
    fn test_ising_model_construction() {
        let mrf = MarkovRandomField::ising_model(3, 3, 0.5, 1.0);
        assert_eq!(mrf.n_nodes, 9);
        assert_eq!(mrf.n_states, 2);
        // 3x3 grid: 3 horizontal + 3 vertical = 6 + 3 = 12 edges
        // Actually: rows=3, cols=3 → right edges = 3*2=6, down edges = 2*3=6 → 12 total
        assert_eq!(mrf.pairwise_potentials.len(), 12);
    }

    #[test]
    fn test_ising_unary_potentials() {
        let h = 0.5;
        let mrf = MarkovRandomField::ising_model(2, 2, h, 1.0);
        // state 0 (spin -1): should have -h
        assert!((mrf.unary_potentials[0][0] - (-h)).abs() < 1e-9);
        // state 1 (spin +1): should have +h
        assert!((mrf.unary_potentials[0][1] - h).abs() < 1e-9);
    }

    #[test]
    fn test_gibbs_sample_count() {
        let mrf = ferromagnetic_ising(5);
        let samples = mrf.gibbs_sample(50, 100, 1, 42).unwrap();
        assert_eq!(samples.len(), 50);
    }

    #[test]
    fn test_gibbs_sample_state_range() {
        let mrf = ferromagnetic_ising(4);
        let samples = mrf.gibbs_sample(100, 50, 1, 7).unwrap();
        for s in &samples {
            assert_eq!(s.len(), 4);
            for &state in s {
                assert!(state < 2, "State {} out of range", state);
            }
        }
    }

    #[test]
    fn test_gibbs_ferromagnetic_bias() {
        // Strong ferromagnetic coupling → most samples should be all-0 or all-1
        let mrf = MarkovRandomField::ising_model(1, 4, 0.0, 5.0);
        let samples = mrf.gibbs_sample(200, 500, 2, 13).unwrap();
        let aligned: usize = samples
            .iter()
            .filter(|s| s.iter().all(|&x| x == s[0]))
            .count();
        let ratio = aligned as f64 / 200.0;
        assert!(
            ratio > 0.7,
            "Expected >70% aligned samples, got {:.0}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_belief_propagation_single_node() {
        // Single node with biased potential
        let mut mrf = MarkovRandomField::new(1, 2);
        // log P ∝ [3, 1] → P ≈ [e^3/(e^3+e^1), e^1/(e^3+e^1)]
        mrf.set_unary(0, vec![3.0, 1.0]).unwrap();
        let beliefs = mrf.belief_propagation(50, 1e-6, 0.0).unwrap();
        let e3 = 3_f64.exp();
        let e1 = 1_f64.exp();
        let expected = e3 / (e3 + e1);
        assert!(
            (beliefs[0][0] - expected).abs() < 1e-4,
            "belief[0]={:.4}, expected={:.4}",
            beliefs[0][0],
            expected
        );
    }

    #[test]
    fn test_belief_propagation_chain_marginals() {
        // Chain of 3 nodes with moderate coupling; beliefs should be proper distributions
        let mrf = ferromagnetic_ising(3);
        let beliefs = mrf.belief_propagation(100, 1e-6, 0.5).unwrap();
        for (i, b) in beliefs.iter().enumerate() {
            let sum: f64 = b.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "beliefs[{}] sum = {}", i, sum);
        }
    }

    #[test]
    fn test_max_product_chain() {
        // Biased Ising chain: h=2 favours spin +1 (state 1)
        let mrf = MarkovRandomField::ising_model(1, 4, 2.0, 0.5);
        let map = mrf.max_product(200, 1e-6).unwrap();
        // All nodes should prefer state 1 under strong external field
        assert_eq!(map.len(), 4);
        let ones = map.iter().filter(|&&s| s == 1).count();
        assert!(ones >= 3, "Expected mostly state 1, got {:?}", map);
    }

    #[test]
    fn test_neighbors() {
        let mut mrf = MarkovRandomField::new(4, 2);
        let pot = array![[0.0, 0.0], [0.0, 0.0]];
        mrf.add_edge(0, 1, pot.clone()).unwrap();
        mrf.add_edge(1, 2, pot.clone()).unwrap();
        mrf.add_edge(2, 3, pot.clone()).unwrap();
        let nb1 = mrf.neighbors(1);
        assert!(nb1.contains(&0));
        assert!(nb1.contains(&2));
    }

    #[test]
    fn test_node_out_of_range_error() {
        let mut mrf = MarkovRandomField::new(3, 2);
        assert!(mrf.set_unary(5, vec![0.0, 0.0]).is_err());
    }
}
