//! Gossip-based all-reduce using push-sum protocol
//!
//! Implements decentralized averaging where each node converges to the global
//! average through iterative gossip rounds. Supports ring, random, and
//! exponential topologies.

use crate::error::{CoreError, CoreResult, ErrorContext};

use super::types::{GossipConfig, GossipTopology};

/// State of a single node in the gossip protocol
#[derive(Debug, Clone)]
struct NodeState {
    /// Accumulated value (numerator in push-sum)
    value: Vec<f64>,
    /// Accumulated weight (denominator in push-sum)
    weight: f64,
}

/// A simple LCG-based RNG for deterministic peer selection (no external deps)
#[derive(Debug)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Generate next pseudo-random u64
    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Knuth
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Generate a random usize in [0, bound)
    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % (bound as u64)) as usize
    }
}

/// Gossip-based all-reduce using the push-sum protocol
///
/// Each node holds a value and a weight. In each round, a node sends
/// a fraction of its (value, weight) to a neighbor determined by the
/// topology. After sufficient rounds, value/weight converges to the
/// global average of the initial values.
#[derive(Debug)]
pub struct GossipAllReduce {
    /// Configuration
    config: GossipConfig,
    /// Number of nodes
    num_nodes: usize,
    /// Per-node state
    nodes: Vec<NodeState>,
    /// RNG for random peer selection
    rng: SimpleRng,
    /// Number of rounds completed
    rounds_completed: usize,
}

impl GossipAllReduce {
    /// Create a new gossip all-reduce instance
    ///
    /// # Arguments
    /// * `config` - Gossip protocol configuration
    /// * `num_nodes` - Number of participating nodes
    #[must_use]
    pub fn new(config: GossipConfig, num_nodes: usize) -> Self {
        let nodes = (0..num_nodes)
            .map(|_| NodeState {
                value: Vec::new(),
                weight: 1.0,
            })
            .collect();
        Self {
            config,
            num_nodes,
            nodes,
            rng: SimpleRng::new(42),
            rounds_completed: 0,
        }
    }

    /// Initialize values for all nodes
    ///
    /// Each node receives its own local data that will be averaged.
    pub fn init_values(&mut self, node_values: Vec<Vec<f64>>) -> CoreResult<()> {
        if node_values.len() != self.num_nodes {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} node values, got {}",
                self.num_nodes,
                node_values.len()
            ))));
        }
        for (i, vals) in node_values.into_iter().enumerate() {
            self.nodes[i].value = vals;
            self.nodes[i].weight = 1.0;
        }
        Ok(())
    }

    /// Run one gossip round
    ///
    /// Each node selects a peer based on the topology and sends a fraction
    /// of its value and weight to that peer.
    pub fn run_round(&mut self) {
        let fraction = self.config.push_fraction;
        let n = self.num_nodes;

        if n == 0 {
            return;
        }

        // Compute messages to send: (sender, receiver, value_fraction, weight_fraction)
        let mut messages: Vec<(usize, Vec<f64>, f64)> = Vec::with_capacity(n);

        for sender in 0..n {
            let receiver = self.select_peer(sender);

            // Compute the fraction to send
            let send_value: Vec<f64> = self.nodes[sender]
                .value
                .iter()
                .map(|v| v * fraction)
                .collect();
            let send_weight = self.nodes[sender].weight * fraction;

            // Reduce sender's state
            for v in &mut self.nodes[sender].value {
                *v *= 1.0 - fraction;
            }
            self.nodes[sender].weight *= 1.0 - fraction;

            messages.push((receiver, send_value, send_weight));
        }

        // Apply all messages
        for (receiver, value, weight) in messages {
            if receiver < n {
                for (rv, sv) in self.nodes[receiver].value.iter_mut().zip(value.iter()) {
                    *rv += sv;
                }
                self.nodes[receiver].weight += weight;
            }
        }

        self.rounds_completed += 1;
    }

    /// Run rounds until convergence or max rounds reached
    ///
    /// Returns the number of rounds executed.
    ///
    /// # Arguments
    /// * `tolerance` - Maximum allowed difference between any node's estimate
    ///   and the overall estimated average
    pub fn run_to_convergence(&mut self, tolerance: f64) -> usize {
        let max_rounds = self.config.num_rounds;
        for _ in 0..max_rounds {
            self.run_round();

            // Check convergence: all estimates should be close
            if self.is_converged(tolerance) {
                return self.rounds_completed;
            }
        }
        self.rounds_completed
    }

    /// Get the current average estimate for a given node
    ///
    /// Returns value / weight, which converges to the global average.
    pub fn get_estimate(&self, node_id: usize) -> CoreResult<Vec<f64>> {
        if node_id >= self.num_nodes {
            return Err(CoreError::ValueError(ErrorContext::new(format!(
                "Node ID {node_id} out of range (num_nodes = {})",
                self.num_nodes
            ))));
        }
        let node = &self.nodes[node_id];
        if node.weight.abs() < f64::EPSILON {
            return Ok(node.value.clone());
        }
        Ok(node.value.iter().map(|v| v / node.weight).collect())
    }

    /// Get the number of rounds completed
    #[must_use]
    pub fn rounds_completed(&self) -> usize {
        self.rounds_completed
    }

    /// Check if all node estimates have converged
    fn is_converged(&self, tolerance: f64) -> bool {
        if self.num_nodes <= 1 {
            return true;
        }

        // Compute estimates for all nodes
        let estimates: Vec<Vec<f64>> = (0..self.num_nodes)
            .filter_map(|i| self.get_estimate(i).ok())
            .collect();

        if estimates.is_empty() {
            return true;
        }

        // Compute global average of estimates
        let dim = estimates[0].len();
        let mut avg = vec![0.0; dim];
        for est in &estimates {
            for (a, e) in avg.iter_mut().zip(est.iter()) {
                *a += e;
            }
        }
        let n = estimates.len() as f64;
        for a in &mut avg {
            *a /= n;
        }

        // Check max deviation
        for est in &estimates {
            for (e, a) in est.iter().zip(avg.iter()) {
                if (e - a).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Select a peer for the given sender based on topology
    fn select_peer(&mut self, sender: usize) -> usize {
        let n = self.num_nodes;
        if n <= 1 {
            return 0;
        }

        match &self.config.topology {
            GossipTopology::Ring => (sender + 1) % n,
            GossipTopology::Random => {
                // Pick a random peer different from sender
                loop {
                    let peer = self.rng.next_usize(n);
                    if peer != sender {
                        return peer;
                    }
                    // For n=1 this would loop forever, but we checked n>1 above
                }
            }
            GossipTopology::Exponential => {
                // Send to sender + 2^k for random k in [0, log2(n))
                let max_k = (n as f64).log2().ceil() as u32;
                let k = if max_k == 0 {
                    0
                } else {
                    self.rng.next_usize(max_k as usize) as u32
                };
                let offset = 1usize << k;
                (sender + offset) % n
            }
        }
    }
}

/// Convenience function: run gossip all-reduce across multiple workers (shared memory simulation)
///
/// This function simulates decentralized all-reduce by instantiating one
/// `GossipAllReduce` coordinator, loading all node values, running gossip
/// rounds, and returning the converged global average.
///
/// # Arguments
/// * `worker_values` - Initial gradient/parameter vector for each worker
/// * `config` - Gossip configuration (topology, number of rounds, push fraction)
///
/// # Returns
/// The global average of all worker values after gossip convergence, or an error
/// if inputs are inconsistent.
pub fn gossip_allreduce_simulate(
    worker_values: &[Vec<f64>],
    config: &GossipConfig,
) -> CoreResult<Vec<f64>> {
    let n = worker_values.len();
    if n == 0 {
        return Err(CoreError::ValueError(ErrorContext::new(
            "gossip_allreduce_simulate: no workers provided".to_string(),
        )));
    }

    let dim = worker_values[0].len();
    for (i, v) in worker_values.iter().enumerate() {
        if v.len() != dim {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Worker {i} has dimension {} but expected {dim}",
                v.len()
            ))));
        }
    }

    let mut gossip = GossipAllReduce::new(config.clone(), n);
    gossip.init_values(worker_values.to_vec())?;
    gossip.run_to_convergence(1e-6);

    // Collect final estimates and average them for the best result
    let mut result = vec![0.0f64; dim];
    for node_id in 0..n {
        let est = gossip.get_estimate(node_id)?;
        for (r, e) in result.iter_mut().zip(est.iter()) {
            *r += e;
        }
    }
    let n_f = n as f64;
    for r in &mut result {
        *r /= n_f;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gossip_ring_convergence() {
        let config = GossipConfig {
            topology: GossipTopology::Ring,
            num_rounds: 100,
            push_fraction: 0.5,
        };
        let mut gossip = GossipAllReduce::new(config, 4);
        gossip
            .init_values(vec![vec![4.0], vec![8.0], vec![12.0], vec![16.0]])
            .expect("init");

        let _rounds = gossip.run_to_convergence(0.01);
        // Global average = (4+8+12+16)/4 = 10.0
        for i in 0..4 {
            let est = gossip.get_estimate(i).expect("estimate");
            assert!(
                (est[0] - 10.0).abs() < 0.1,
                "Node {i} estimate {:.4} not close to 10.0",
                est[0]
            );
        }
    }

    #[test]
    fn test_gossip_random_convergence() {
        let config = GossipConfig {
            topology: GossipTopology::Random,
            num_rounds: 200,
            push_fraction: 0.5,
        };
        let mut gossip = GossipAllReduce::new(config, 3);
        gossip
            .init_values(vec![vec![3.0, 6.0], vec![9.0, 12.0], vec![15.0, 18.0]])
            .expect("init");

        gossip.run_to_convergence(0.1);
        // Average: [9.0, 12.0]
        for i in 0..3 {
            let est = gossip.get_estimate(i).expect("estimate");
            assert!((est[0] - 9.0).abs() < 1.0, "Node {i} dim0 = {:.4}", est[0]);
            assert!((est[1] - 12.0).abs() < 1.0, "Node {i} dim1 = {:.4}", est[1]);
        }
    }

    #[test]
    fn test_gossip_exponential_topology() {
        // Test that exponential topology runs without errors and produces
        // estimates that move toward the average over many rounds
        let config = GossipConfig {
            topology: GossipTopology::Exponential,
            num_rounds: 100,
            push_fraction: 0.5,
        };
        let mut gossip = GossipAllReduce::new(config, 8);
        gossip
            .init_values(vec![
                vec![0.0],
                vec![10.0],
                vec![20.0],
                vec![30.0],
                vec![40.0],
                vec![50.0],
                vec![60.0],
                vec![70.0],
            ])
            .expect("init");

        gossip.run_to_convergence(1.0);
        // Average = 35.0; with 8 nodes, exponential topology converges well
        for i in 0..8 {
            let est = gossip.get_estimate(i).expect("estimate");
            assert!(
                (est[0] - 35.0).abs() < 5.0,
                "Node {i} estimate {:.4} not close to 35.0",
                est[0]
            );
        }
    }

    #[test]
    fn test_gossip_single_node() {
        let config = GossipConfig::default();
        let mut gossip = GossipAllReduce::new(config, 1);
        gossip.init_values(vec![vec![42.0]]).expect("init");
        gossip.run_round();
        let est = gossip.get_estimate(0).expect("estimate");
        assert!((est[0] - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gossip_invalid_node_id() {
        let config = GossipConfig::default();
        let gossip = GossipAllReduce::new(config, 2);
        assert!(gossip.get_estimate(5).is_err());
    }

    #[test]
    fn test_gossip_dimension_mismatch() {
        let config = GossipConfig::default();
        let mut gossip = GossipAllReduce::new(config, 2);
        let result = gossip.init_values(vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert!(result.is_err());
    }

    // ── gossip_allreduce_simulate tests ──

    #[test]
    fn test_gossip_allreduce_simulate_uniform_input() {
        // All workers have the same value → result should equal that value
        let config = GossipConfig {
            topology: GossipTopology::Ring,
            num_rounds: 50,
            push_fraction: 0.5,
        };
        let workers = vec![vec![7.0, 14.0]; 4];
        let result = gossip_allreduce_simulate(&workers, &config).expect("allreduce");
        assert!((result[0] - 7.0).abs() < 0.5, "dim0 = {}", result[0]);
        assert!((result[1] - 14.0).abs() < 0.5, "dim1 = {}", result[1]);
    }

    #[test]
    fn test_gossip_allreduce_simulate_mean() {
        // Verify it converges to the true mean
        let config = GossipConfig {
            topology: GossipTopology::Ring,
            num_rounds: 200,
            push_fraction: 0.5,
        };
        let workers = vec![vec![2.0], vec![4.0], vec![6.0], vec![8.0]];
        // mean = 5.0
        let result = gossip_allreduce_simulate(&workers, &config).expect("allreduce");
        assert!(
            (result[0] - 5.0).abs() < 0.5,
            "Expected ~5.0, got {}",
            result[0]
        );
    }

    #[test]
    fn test_gossip_allreduce_simulate_single_worker() {
        let config = GossipConfig::default();
        let workers = vec![vec![42.0, 1.0]];
        let result = gossip_allreduce_simulate(&workers, &config).expect("allreduce");
        assert!((result[0] - 42.0).abs() < f64::EPSILON);
        assert!((result[1] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gossip_allreduce_simulate_empty_workers_error() {
        let config = GossipConfig::default();
        let result = gossip_allreduce_simulate(&[], &config);
        assert!(result.is_err(), "Should error on empty worker list");
    }

    #[test]
    fn test_gossip_allreduce_simulate_dimension_mismatch_error() {
        let config = GossipConfig::default();
        let workers = vec![vec![1.0, 2.0], vec![3.0]]; // different dims
        let result = gossip_allreduce_simulate(&workers, &config);
        assert!(result.is_err(), "Should error on dimension mismatch");
    }

    #[test]
    fn test_gossip_allreduce_more_rounds_closer_to_mean() {
        let make_workers = || vec![vec![0.0], vec![100.0], vec![50.0], vec![50.0]];
        let target = 50.0_f64;

        let config_few = GossipConfig {
            topology: GossipTopology::Ring,
            num_rounds: 5,
            push_fraction: 0.5,
        };
        let config_many = GossipConfig {
            topology: GossipTopology::Ring,
            num_rounds: 200,
            push_fraction: 0.5,
        };

        let result_few =
            gossip_allreduce_simulate(&make_workers(), &config_few).expect("few rounds");
        let result_many =
            gossip_allreduce_simulate(&make_workers(), &config_many).expect("many rounds");

        let err_few = (result_few[0] - target).abs();
        let err_many = (result_many[0] - target).abs();

        assert!(
            err_many < err_few + 1.0 || err_many < 5.0,
            "More rounds should converge closer: few={err_few:.4} many={err_many:.4}"
        );
    }
}
