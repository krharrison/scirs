//! Approximate inference algorithms for Bayesian Networks.
//!
//! Provides:
//! - [`GibbsSampler`] — Markov Chain Monte Carlo sampling
//! - [`LikelihoodWeighting`] — importance sampling
//! - [`MeanFieldVI`] — mean-field variational inference

use std::collections::HashMap;
use std::sync::Arc;
use crate::StatsError;
use super::{
    dag::DAG,
    cpd::CPD,
    exact_inference::BayesianNetwork,
};

// ---------------------------------------------------------------------------
// Simple pseudo-random generator (no rand dep — uses LCG)
// ---------------------------------------------------------------------------

/// A minimal pseudo-random number generator trait for approximate inference.
pub trait Rng {
    /// Sample a uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64;

    /// Sample an integer in `[0, n)`.
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_f64() * n as f64) as usize
    }
}

/// LCG-based pseudo-random number generator (standalone, no external deps).
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
}

impl Rng for LcgRng {
    fn next_f64(&mut self) -> f64 {
        self.state = self.state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Take the top 53 bits for a uniform double
        let bits = (self.state >> 11) | 0x3FF0_0000_0000_0000u64;
        let f = f64::from_bits(bits) - 1.0;
        f.clamp(0.0, 1.0 - f64::EPSILON)
    }
}

/// Sample a categorical distribution given unnormalised probabilities.
fn sample_categorical(probs: &[f64], rng: &mut impl Rng) -> usize {
    let sum: f64 = probs.iter().sum();
    let u = rng.next_f64() * sum;
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i;
        }
    }
    probs.len() - 1 // fallback
}

// ---------------------------------------------------------------------------
// GibbsSampler
// ---------------------------------------------------------------------------

/// Gibbs sampler for approximate inference in Bayesian Networks.
///
/// Iterates over all non-evidence variables and samples each from its
/// conditional distribution P(X_i | Markov_blanket(X_i)).
pub struct GibbsSampler {
    /// The Bayesian Network.
    pub bn: Arc<BayesianNetwork>,
    /// Number of samples to collect.
    pub n_samples: usize,
    /// Number of burn-in samples to discard.
    pub burn_in: usize,
}

impl GibbsSampler {
    /// Create a new GibbsSampler.
    pub fn new(bn: Arc<BayesianNetwork>, n_samples: usize, burn_in: usize) -> Self {
        Self { bn, n_samples, burn_in }
    }

    /// Run the Gibbs sampler and return collected samples.
    ///
    /// Each sample is a complete assignment to all nodes.
    pub fn sample(
        &self,
        evidence: &HashMap<usize, usize>,
        rng: &mut impl Rng,
    ) -> Result<Vec<Vec<usize>>, StatsError> {
        let n = self.bn.dag.n_nodes;
        // Free variables (non-evidence)
        let free_vars: Vec<usize> = (0..n)
            .filter(|v| !evidence.contains_key(v))
            .collect();
        if free_vars.is_empty() {
            return Ok(Vec::new());
        }

        // Initialize state: evidence vars fixed, others randomly
        let mut state: Vec<usize> = (0..n).map(|v| {
            if let Some(&val) = evidence.get(&v) {
                val
            } else {
                let card = self.bn.cpds[v].cardinality();
                if card == 0 { 0 } else { rng.next_usize(card) }
            }
        }).collect();

        let total = self.burn_in + self.n_samples;
        let mut samples = Vec::with_capacity(self.n_samples);

        for iter in 0..total {
            // Update each free variable
            for &v in &free_vars {
                let cond_dist = self.compute_conditional(v, &state)?;
                state[v] = sample_categorical(&cond_dist, rng);
            }
            if iter >= self.burn_in {
                samples.push(state.clone());
            }
        }
        Ok(samples)
    }

    /// Query: P(query_var | evidence) via Gibbs sampling.
    pub fn query(
        &self,
        query_var: usize,
        evidence: &HashMap<usize, usize>,
        rng: &mut impl Rng,
    ) -> Result<Vec<f64>, StatsError> {
        let card = self.bn.cpds[query_var].cardinality();
        if card == 0 {
            return Err(StatsError::InvalidInput(format!(
                "Node {query_var} is continuous; use density estimation instead"
            )));
        }
        let samples = self.sample(evidence, rng)?;
        let mut counts = vec![0usize; card];
        for sample in &samples {
            counts[sample[query_var]] += 1;
        }
        let total = samples.len() as f64;
        let mut probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();
        // Normalize
        let s: f64 = probs.iter().sum();
        if s > 1e-300 {
            for p in &mut probs {
                *p /= s;
            }
        }
        Ok(probs)
    }

    /// Compute P(X_v = val | state[v] for all v != v_idx) using Markov blanket.
    fn compute_conditional(
        &self,
        v: usize,
        state: &[usize],
    ) -> Result<Vec<f64>, StatsError> {
        let card = self.bn.cpds[v].cardinality();
        if card == 0 {
            return Err(StatsError::InvalidInput(format!(
                "Node {v} is continuous; Gibbs sampling requires discrete nodes"
            )));
        }
        let dag = &self.bn.dag;
        let mut probs = vec![0.0f64; card];
        for val in 0..card {
            let mut log_prob = 0.0f64;
            // P(X_v = val | pa(X_v))
            let pa: Vec<usize> = dag.parents[v].iter().map(|&p| state[p]).collect();
            let p = self.bn.cpds[v].prob(val, &pa);
            if p <= 0.0 { probs[val] = 0.0; continue; }
            log_prob += p.ln();
            // P(X_ch = state[ch] | pa(X_ch)) for each child ch
            for &ch in &dag.children[v] {
                let ch_pa: Vec<usize> = dag.parents[ch].iter().map(|&p| {
                    if p == v { val } else { state[p] }
                }).collect();
                let p_ch = self.bn.cpds[ch].prob(state[ch], &ch_pa);
                if p_ch <= 0.0 { log_prob = f64::NEG_INFINITY; break; }
                log_prob += p_ch.ln();
            }
            probs[val] = log_prob.exp();
        }
        Ok(probs)
    }
}

// ---------------------------------------------------------------------------
// LikelihoodWeighting
// ---------------------------------------------------------------------------

/// Likelihood Weighting for approximate inference.
///
/// Samples from the prior, weighting each sample by the probability of
/// the observed evidence.
pub struct LikelihoodWeighting {
    /// Number of samples to generate.
    pub n_samples: usize,
}

impl LikelihoodWeighting {
    /// Create a new LikelihoodWeighting sampler.
    pub fn new(n_samples: usize) -> Self {
        Self { n_samples }
    }

    /// Query P(query_var | evidence) using likelihood weighting.
    pub fn query(
        &self,
        bn: &BayesianNetwork,
        query_var: usize,
        evidence: &HashMap<usize, usize>,
        rng: &mut impl Rng,
    ) -> Result<Vec<f64>, StatsError> {
        let card = bn.cpds[query_var].cardinality();
        if card == 0 {
            return Err(StatsError::InvalidInput(format!(
                "Node {query_var} is continuous"
            )));
        }
        let topo = bn.dag.topological_sort();
        let mut weighted_counts = vec![0.0f64; card];

        for _ in 0..self.n_samples {
            let (sample, weight) = self.generate_weighted_sample(bn, &topo, evidence, rng)?;
            weighted_counts[sample[query_var]] += weight;
        }
        let total: f64 = weighted_counts.iter().sum();
        if total < 1e-300 {
            // All weights collapsed — return uniform
            let card_f = card as f64;
            return Ok(vec![1.0 / card_f; card]);
        }
        Ok(weighted_counts.iter().map(|&c| c / total).collect())
    }

    /// Generate one weighted sample.
    fn generate_weighted_sample(
        &self,
        bn: &BayesianNetwork,
        topo: &[usize],
        evidence: &HashMap<usize, usize>,
        rng: &mut impl Rng,
    ) -> Result<(Vec<usize>, f64), StatsError> {
        let n = bn.dag.n_nodes;
        let mut sample = vec![0usize; n];
        let mut log_weight = 0.0f64;

        for &node in topo {
            let cpd = &bn.cpds[node];
            let pa: Vec<usize> = bn.dag.parents[node].iter().map(|&p| sample[p]).collect();
            if let Some(&obs_val) = evidence.get(&node) {
                // Observed: fix value and accumulate weight
                sample[node] = obs_val;
                let p = cpd.prob(obs_val, &pa);
                if p <= 0.0 {
                    log_weight = f64::NEG_INFINITY;
                } else {
                    log_weight += p.ln();
                }
            } else {
                // Unobserved: sample from CPD
                let card = cpd.cardinality();
                if card == 0 {
                    sample[node] = 0; // continuous: skip
                } else {
                    let probs: Vec<f64> = (0..card).map(|v| cpd.prob(v, &pa)).collect();
                    sample[node] = sample_categorical(&probs, rng);
                }
            }
        }
        Ok((sample, log_weight.exp()))
    }
}

// ---------------------------------------------------------------------------
// MeanFieldVI
// ---------------------------------------------------------------------------

/// Mean-field variational inference for discrete Bayesian Networks.
///
/// Approximates the posterior P(X | evidence) with a fully factored distribution
/// q(X) = ∏_i q_i(X_i), optimized by coordinate ascent variational inference (CAVI).
pub struct MeanFieldVI {
    /// Maximum number of CAVI iterations.
    pub max_iter: usize,
    /// Convergence tolerance (ELBO change).
    pub tol: f64,
}

impl Default for MeanFieldVI {
    fn default() -> Self {
        Self { max_iter: 100, tol: 1e-6 }
    }
}

impl MeanFieldVI {
    /// Create a new MeanFieldVI instance.
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Run mean-field VI. Returns approximate posterior marginals q_i for each node.
    ///
    /// `q[i]` = distribution q_i(X_i) over the cardinality of node i.
    /// Evidence nodes are fixed.
    pub fn run(
        &self,
        bn: &BayesianNetwork,
        evidence: &HashMap<usize, usize>,
    ) -> Result<Vec<Vec<f64>>, StatsError> {
        let n = bn.dag.n_nodes;

        // Initialize q_i uniformly (or as point mass for evidence nodes)
        let mut q: Vec<Vec<f64>> = (0..n).map(|i| {
            let card = bn.cpds[i].cardinality();
            if card == 0 {
                return vec![1.0];
            }
            if let Some(&val) = evidence.get(&i) {
                let mut v = vec![0.0; card];
                v[val] = 1.0;
                v
            } else {
                vec![1.0 / card as f64; card]
            }
        }).collect();

        let topo = bn.dag.topological_sort();

        for _iter in 0..self.max_iter {
            let old_q = q.clone();

            // CAVI update for each free variable
            for &node in &topo {
                if evidence.contains_key(&node) {
                    continue;
                }
                let card = bn.cpds[node].cardinality();
                if card == 0 {
                    continue;
                }
                // Update q_node ∝ exp(E_q[log P(node | pa(node))] + Σ_{ch} E_q[log P(ch | pa(ch))])
                let mut log_q = vec![0.0f64; card];
                for val in 0..card {
                    log_q[val] += self.expected_log_cpd(bn, node, val, &q);
                    // Add contribution from children
                    for &ch in &bn.dag.children[node] {
                        log_q[val] += self.expected_log_child(bn, ch, node, val, &q);
                    }
                }
                // Softmax
                let max_l = log_q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = log_q.iter().map(|&l| (l - max_l).exp()).collect();
                let sum: f64 = exps.iter().sum();
                q[node] = if sum > 1e-300 {
                    exps.iter().map(|e| e / sum).collect()
                } else {
                    vec![1.0 / card as f64; card]
                };
            }

            // Check convergence (max absolute change in q)
            let max_change = q.iter().zip(&old_q)
                .flat_map(|(qi, qi_old)| {
                    qi.iter().zip(qi_old).map(|(&a, &b)| (a - b).abs())
                })
                .fold(0.0f64, f64::max);
            if max_change < self.tol {
                break;
            }
        }
        Ok(q)
    }

    /// E_q[log P(node=val | pa(node))], averaging over parent distributions.
    fn expected_log_cpd(
        &self,
        bn: &BayesianNetwork,
        node: usize,
        val: usize,
        q: &[Vec<f64>],
    ) -> f64 {
        let cpd = &bn.cpds[node];
        let parents = &bn.dag.parents[node];
        if parents.is_empty() {
            let p = cpd.prob(val, &[]);
            return if p > 0.0 { p.ln() } else { -1e10 };
        }
        // Enumerate all parent configurations, weighting by product of q
        let parent_cards: Vec<usize> = parents.iter().map(|&p| q[p].len()).collect();
        let n_configs: usize = parent_cards.iter().product();
        let mut expected = 0.0f64;
        for config_idx in 0..n_configs {
            let pa_vals = decode_config(config_idx, &parent_cards);
            let weight: f64 = parents.iter().zip(&pa_vals)
                .map(|(&p, &pv)| q[p][pv])
                .product();
            let p = cpd.prob(val, &pa_vals);
            let log_p = if p > 0.0 { p.ln() } else { -1e10 };
            expected += weight * log_p;
        }
        expected
    }

    /// E_q[log P(child=ch_val | pa(child))], where node=`node` takes value `val`
    /// and other parents are averaged over q.
    fn expected_log_child(
        &self,
        bn: &BayesianNetwork,
        child: usize,
        node: usize,
        node_val: usize,
        q: &[Vec<f64>],
    ) -> f64 {
        let cpd = &bn.cpds[child];
        let ch_card = cpd.cardinality();
        if ch_card == 0 { return 0.0; }
        let parents = &bn.dag.children[node]; // This is wrong — we want child's parents
        let ch_parents = &bn.dag.parents[child];
        // Other parents of child (excluding `node`)
        let other_parents: Vec<usize> = ch_parents.iter()
            .copied()
            .filter(|&p| p != node)
            .collect();
        let other_cards: Vec<usize> = other_parents.iter().map(|&p| q[p].len()).collect();
        let n_configs: usize = if other_cards.is_empty() { 1 } else { other_cards.iter().product() };
        let mut expected = 0.0f64;
        for config_idx in 0..n_configs {
            let other_vals = decode_config(config_idx, &other_cards);
            let weight: f64 = other_parents.iter().zip(&other_vals)
                .map(|(&p, &pv)| q[p][pv])
                .product::<f64>();
            // Build full parent assignment for child
            let pa_vals: Vec<usize> = ch_parents.iter().map(|&p| {
                if p == node { node_val }
                else {
                    let pos = other_parents.iter().position(|&op| op == p).unwrap_or(0);
                    other_vals[pos]
                }
            }).collect();
            // Sum over child values (marginalize)
            let mut child_expected = 0.0f64;
            for ch_val in 0..ch_card {
                let q_ch = q[child][ch_val];
                let p = cpd.prob(ch_val, &pa_vals);
                let log_p = if p > 0.0 { p.ln() } else { -1e10 };
                child_expected += q_ch * log_p;
            }
            expected += weight * child_expected;
        }
        let _ = parents; // used indirectly
        expected
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn decode_config(mut idx: usize, cards: &[usize]) -> Vec<usize> {
    let n = cards.len();
    let mut result = vec![0usize; n];
    for i in (0..n).rev() {
        if cards[i] == 0 {
            continue;
        }
        result[i] = idx % cards[i];
        idx /= cards[i];
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
    use crate::bayesian_network::exact_inference::BayesianNetwork;

    fn wet_grass_network() -> Arc<BayesianNetwork> {
        let mut dag = DAG::new(3);
        dag.add_edge(0, 2).unwrap();
        dag.add_edge(1, 2).unwrap();
        let cpd_rain = TabularCPD::new(0, 2, vec![], vec![], vec![vec![0.8, 0.2]]).unwrap();
        let cpd_spr  = TabularCPD::new(1, 2, vec![], vec![], vec![vec![0.5, 0.5]]).unwrap();
        let cpd_wg   = TabularCPD::new(2, 2, vec![0, 1], vec![2, 2], vec![
            vec![0.99, 0.01], vec![0.01, 0.99],
            vec![0.01, 0.99], vec![0.01, 0.99],
        ]).unwrap();
        let cpds: Vec<Box<dyn CPD>> = vec![
            Box::new(cpd_rain), Box::new(cpd_spr), Box::new(cpd_wg),
        ];
        Arc::new(BayesianNetwork::new(dag, cpds).unwrap())
    }

    #[test]
    fn test_gibbs_prior_rain() {
        let bn = wet_grass_network();
        let sampler = GibbsSampler::new(Arc::clone(&bn), 5000, 500);
        let mut rng = LcgRng::new(42);
        let probs = sampler.query(0, &HashMap::new(), &mut rng).unwrap();
        // P(Rain=0) ≈ 0.8 with tolerance
        assert!((probs[0] - 0.8).abs() < 0.05, "P(Rain=0) ≈ 0.8, got {}", probs[0]);
    }

    #[test]
    fn test_likelihood_weighting_prior() {
        let bn = wet_grass_network();
        let lw = LikelihoodWeighting::new(5000);
        let mut rng = LcgRng::new(42);
        let probs = lw.query(&bn, 0, &HashMap::new(), &mut rng).unwrap();
        assert!((probs[0] - 0.8).abs() < 0.05, "P(Rain=0) ≈ 0.8, got {}", probs[0]);
    }

    #[test]
    fn test_likelihood_weighting_conditional() {
        let bn = wet_grass_network();
        let lw = LikelihoodWeighting::new(5000);
        let mut rng = LcgRng::new(99);
        let mut evidence = HashMap::new();
        evidence.insert(2usize, 1usize); // WetGrass = 1
        let probs = lw.query(&bn, 0, &evidence, &mut rng).unwrap();
        // P(Rain=1 | WG=1) should be higher than prior
        assert!(probs[1] > 0.2, "P(Rain=1|WG=1) should be > 0.2, got {}", probs[1]);
    }

    #[test]
    fn test_mean_field_prior() {
        let bn = wet_grass_network();
        let mf = MeanFieldVI::default();
        let q = mf.run(&bn, &HashMap::new()).unwrap();
        // Rain prior should be approximately 0.8
        assert!((q[0][0] - 0.8).abs() < 0.1, "q(Rain=0) ≈ 0.8, got {}", q[0][0]);
    }

    #[test]
    fn test_mean_field_with_evidence() {
        let bn = wet_grass_network();
        let mf = MeanFieldVI::default();
        let mut evidence = HashMap::new();
        evidence.insert(2usize, 1usize); // WetGrass = 1
        let q = mf.run(&bn, &evidence).unwrap();
        // Evidence node should be fixed
        assert!((q[2][1] - 1.0).abs() < 1e-9);
    }
}
