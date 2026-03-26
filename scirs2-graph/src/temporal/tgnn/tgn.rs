//! Temporal Graph Network (TGN).
//!
//! Implements the TGN model from:
//! > Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020).
//! > "Temporal Graph Networks for Deep Learning on Dynamic Graphs."
//! > ICML 2020 Workshop. <https://arxiv.org/abs/2006.10637>
//!
//! ## Architecture
//!
//! TGN maintains a **per-node memory** `s_i ∈ R^{memory_dim}` that is updated
//! as interactions arrive:
//!
//! 1. **Message Module**: Given an interaction `(i, j, t, e_ij)`:
//!    ```text
//!    m_i = msg(s_i, s_j, Δt, e_ij) = Linear(concat(s_i, s_j, e_ij, φ(Δt)))
//!    ```
//! 2. **Memory Updater** (GRU):
//!    ```text
//!    s_i(t) = GRU(s_i(t⁻), m_i)
//!    ```
//! 3. **Embedding Module** (temporal attention, like TGAT):
//!    ```text
//!    z_i(t) = TempAttn(s_i(t), {s_j(t_ij)}_{j∈N_t(i)}, t)
//!    ```

use super::time_encoding::{
    concat, matvec, relu_vec, scaled_dot_product, sigmoid_vec, softmax, tanh_vec, xavier_init,
    TimeEncode,
};
use super::types::{TgnnEdge, TgnnGraph, TgnConfig, TemporalPrediction};
use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// TgnMemory
// ─────────────────────────────────────────────────────────────────────────────

/// Per-node memory state for TGN.
///
/// Maintains a `memory_dim`-dimensional vector for each node, plus the
/// timestamp of the last update.
#[derive(Debug, Clone)]
pub struct TgnMemory {
    /// Memory vectors, shape `[n_nodes][memory_dim]`
    pub state: Vec<Vec<f64>>,
    /// Timestamp of last update for each node
    pub last_update: Vec<f64>,
    /// Memory dimension
    pub memory_dim: usize,
}

impl TgnMemory {
    /// Create a zero-initialised memory for `n_nodes` nodes.
    pub fn new(n_nodes: usize, memory_dim: usize) -> Self {
        TgnMemory {
            state: vec![vec![0.0f64; memory_dim]; n_nodes],
            last_update: vec![0.0f64; n_nodes],
            memory_dim,
        }
    }

    /// Get the memory of node `i`.
    pub fn get(&self, i: usize) -> &[f64] {
        self.state.get(i).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Set the memory of node `i`.
    pub fn set(&mut self, i: usize, new_state: Vec<f64>, timestamp: f64) -> Result<()> {
        if i >= self.state.len() {
            return Err(GraphError::InvalidParameter {
                param: "node_index".to_string(),
                value: format!("{}", i),
                expected: format!("< {}", self.state.len()),
                context: "TgnMemory::set".to_string(),
            });
        }
        if new_state.len() != self.memory_dim {
            return Err(GraphError::InvalidParameter {
                param: "new_state.len".to_string(),
                value: format!("{}", new_state.len()),
                expected: format!("{}", self.memory_dim),
                context: "TgnMemory::set".to_string(),
            });
        }
        self.state[i] = new_state;
        self.last_update[i] = timestamp;
        Ok(())
    }

    /// Reset all memories to zero.
    pub fn reset(&mut self) {
        for s in &mut self.state {
            s.iter_mut().for_each(|x| *x = 0.0);
        }
        self.last_update.iter_mut().for_each(|t| *t = 0.0);
    }

    /// Number of nodes
    pub fn n_nodes(&self) -> usize {
        self.state.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TgnMessageModule
// ─────────────────────────────────────────────────────────────────────────────

/// Computes messages from source to destination and vice versa.
///
/// The message function is:
/// ```text
/// m = ReLU(W_msg · concat(s_src, s_dst, e_feat, φ(Δt)) + b_msg)
/// ```
/// where Δt is the time since the node's last interaction.
#[derive(Debug, Clone)]
pub struct TgnMessageModule {
    /// Linear layer weights (message_dim × input_dim)
    w_msg: Vec<Vec<f64>>,
    /// Bias vector
    b_msg: Vec<f64>,
    /// Output message dimension
    pub message_dim: usize,
    /// Time encoder
    time_enc: TimeEncode,
}

impl TgnMessageModule {
    /// Create a new message module.
    ///
    /// `memory_dim` + `memory_dim` + `edge_feat_dim` + `time_dim` = input_dim.
    pub fn new(
        memory_dim: usize,
        edge_feat_dim: usize,
        time_dim: usize,
        message_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        let time_enc = TimeEncode::new(time_dim)?;
        let input_dim = 2 * memory_dim + edge_feat_dim + time_dim;
        let w_msg = xavier_init(message_dim, input_dim, seed);
        let b_msg = vec![0.0f64; message_dim];
        Ok(TgnMessageModule {
            w_msg,
            b_msg,
            message_dim,
            time_enc,
        })
    }

    /// Compute message from `src` to `dst` at time `t`.
    ///
    /// `s_src` and `s_dst` are their current memory states.
    /// `edge_feat` is the edge feature vector (may be empty, padded with zeros).
    /// `delta_t` is the time elapsed since the src node's last update.
    pub fn compute(
        &self,
        s_src: &[f64],
        s_dst: &[f64],
        edge_feat: &[f64],
        delta_t: f64,
    ) -> Vec<f64> {
        let phi = self.time_enc.encode(delta_t);

        // Determine expected edge_feat size from w_msg width
        let total_input = self.w_msg[0].len();
        let expected_edge_dim = total_input - 2 * s_src.len() - phi.len();

        let edge_padded: Vec<f64> = if edge_feat.len() >= expected_edge_dim {
            edge_feat[..expected_edge_dim].to_vec()
        } else {
            let mut v = edge_feat.to_vec();
            v.resize(expected_edge_dim, 0.0);
            v
        };

        let input = concat(&concat(s_src, s_dst), &concat(&edge_padded, &phi));
        let raw = matvec(&self.w_msg, &input);
        let with_bias: Vec<f64> = raw.iter().zip(self.b_msg.iter()).map(|(r, b)| r + b).collect();
        relu_vec(&with_bias)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TgnMemoryUpdater (GRU)
// ─────────────────────────────────────────────────────────────────────────────

/// GRU-style memory updater.
///
/// Given previous memory `h` and incoming message `x`:
/// ```text
/// z = σ(W_z · concat(h, x) + b_z)   # update gate
/// r = σ(W_r · concat(h, x) + b_r)   # reset gate
/// h̃ = tanh(W_h · concat(r ∗ h, x) + b_h)   # candidate
/// h_new = (1 - z) ∗ h + z ∗ h̃
/// ```
#[derive(Debug, Clone)]
pub struct TgnMemoryUpdater {
    /// Update gate weights (memory_dim × (memory_dim + message_dim))
    w_z: Vec<Vec<f64>>,
    b_z: Vec<f64>,
    /// Reset gate weights
    w_r: Vec<Vec<f64>>,
    b_r: Vec<f64>,
    /// Candidate state weights (memory_dim × (memory_dim + message_dim))
    w_h: Vec<Vec<f64>>,
    b_h: Vec<f64>,
    /// Memory dimension
    pub memory_dim: usize,
}

impl TgnMemoryUpdater {
    /// Create a GRU memory updater.
    pub fn new(memory_dim: usize, message_dim: usize, seed: u64) -> Self {
        let input_dim = memory_dim + message_dim;
        let w_z = xavier_init(memory_dim, input_dim, seed);
        let b_z = vec![0.0f64; memory_dim];
        let w_r = xavier_init(memory_dim, input_dim, seed.wrapping_add(100));
        let b_r = vec![0.0f64; memory_dim];
        let w_h = xavier_init(memory_dim, input_dim, seed.wrapping_add(200));
        let b_h = vec![0.0f64; memory_dim];
        TgnMemoryUpdater {
            w_z,
            b_z,
            w_r,
            b_r,
            w_h,
            b_h,
            memory_dim,
        }
    }

    /// Apply one GRU step: given previous state `h` and message `x`, return new state.
    pub fn step(&self, h: &[f64], x: &[f64]) -> Vec<f64> {
        let hx = concat(h, x);

        // Update gate
        let z_raw = matvec(&self.w_z, &hx);
        let z_with_bias: Vec<f64> = z_raw.iter().zip(self.b_z.iter()).map(|(r, b)| r + b).collect();
        let z = sigmoid_vec(&z_with_bias);

        // Reset gate
        let r_raw = matvec(&self.w_r, &hx);
        let r_with_bias: Vec<f64> = r_raw.iter().zip(self.b_r.iter()).map(|(r, b)| r + b).collect();
        let r = sigmoid_vec(&r_with_bias);

        // Reset: r ∗ h (element-wise)
        let r_h: Vec<f64> = r.iter().zip(h.iter()).map(|(ri, hi)| ri * hi).collect();
        let r_hx = concat(&r_h, x);

        // Candidate
        let h_raw = matvec(&self.w_h, &r_hx);
        let h_with_bias: Vec<f64> = h_raw.iter().zip(self.b_h.iter()).map(|(r, b)| r + b).collect();
        let h_tilde = tanh_vec(&h_with_bias);

        // Final state: (1 - z) * h + z * h_tilde
        h_tilde
            .iter()
            .zip(h.iter())
            .zip(z.iter())
            .map(|((ht, hi), zi)| (1.0 - zi) * hi + zi * ht)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TgnEmbedding (temporal attention like TGAT but using memory)
// ─────────────────────────────────────────────────────────────────────────────

/// Temporal attention embedding module for TGN.
///
/// Computes a node embedding at time `t` using its current memory
/// and the memories of its temporal neighbors.
#[derive(Debug, Clone)]
pub struct TgnEmbedding {
    /// Query projection (embedding_dim × memory_dim + time_dim)
    w_q: Vec<Vec<f64>>,
    /// Key projection (embedding_dim × memory_dim + time_dim)
    w_k: Vec<Vec<f64>>,
    /// Value projection (embedding_dim × memory_dim + time_dim)
    w_v: Vec<Vec<f64>>,
    /// Output projection (embedding_dim × embedding_dim)
    w_o: Vec<Vec<f64>>,
    b_o: Vec<f64>,
    /// Time encoder
    time_enc: TimeEncode,
    /// Embedding output dimension
    pub embedding_dim: usize,
}

impl TgnEmbedding {
    /// Create the embedding module.
    pub fn new(memory_dim: usize, time_dim: usize, embedding_dim: usize, seed: u64) -> Result<Self> {
        let time_enc = TimeEncode::new(time_dim)?;
        let input_dim = memory_dim + time_dim;
        let w_q = xavier_init(embedding_dim, input_dim, seed);
        let w_k = xavier_init(embedding_dim, input_dim, seed.wrapping_add(10));
        let w_v = xavier_init(embedding_dim, input_dim, seed.wrapping_add(20));
        let w_o = xavier_init(embedding_dim, embedding_dim, seed.wrapping_add(30));
        let b_o = vec![0.0f64; embedding_dim];
        Ok(TgnEmbedding {
            w_q,
            w_k,
            w_v,
            w_o,
            b_o,
            time_enc,
            embedding_dim,
        })
    }

    /// Compute embedding for one node.
    ///
    /// `s_self` is the current memory of the node.
    /// `neighbors` is a list of `(s_neighbor, t_interaction)` — temporal neighbors
    ///   with their interaction timestamps.
    /// `query_time` is the time at which we compute the embedding.
    pub fn embed_node(&self, s_self: &[f64], neighbors: &[(&[f64], f64)], query_time: f64) -> Vec<f64> {
        // Query: self memory + time encoding at Δt=0
        let phi_self = self.time_enc.encode(0.0);
        let q_input = concat(s_self, &phi_self);
        let q = matvec(&self.w_q, &q_input);

        if neighbors.is_empty() {
            // No neighbors: use self-loop (attend over own memory)
            let v_self = matvec(&self.w_v, &q_input);
            let out = matvec(&self.w_o, &v_self);
            let with_bias: Vec<f64> = out.iter().zip(self.b_o.iter()).map(|(o, b)| o + b).collect();
            return relu_vec(&with_bias);
        }

        // Build key and value inputs for each neighbor
        let kv_inputs: Vec<Vec<f64>> = neighbors
            .iter()
            .map(|(s_nbr, t_nbr)| {
                let phi = self.time_enc.encode_delta(query_time, *t_nbr);
                concat(s_nbr, &phi)
            })
            .collect();

        let keys: Vec<Vec<f64>> = kv_inputs.iter().map(|kv| matvec(&self.w_k, kv)).collect();
        let values: Vec<Vec<f64>> = kv_inputs.iter().map(|kv| matvec(&self.w_v, kv)).collect();

        let logits = scaled_dot_product(&q, &keys);
        let alphas = softmax(&logits);

        let mut attended = vec![0.0f64; self.embedding_dim];
        for (alpha, val) in alphas.iter().zip(values.iter()) {
            for (a, v) in attended.iter_mut().zip(val.iter()) {
                *a += alpha * v;
            }
        }

        let out = matvec(&self.w_o, &attended);
        let with_bias: Vec<f64> = out.iter().zip(self.b_o.iter()).map(|(o, b)| o + b).collect();
        relu_vec(&with_bias)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TgnModel
// ─────────────────────────────────────────────────────────────────────────────

/// Full Temporal Graph Network model.
///
/// Maintains per-node memories updated as events arrive, and computes
/// temporal attention embeddings at arbitrary query times.
///
/// ## Usage
///
/// ```rust,no_run
/// use scirs2_graph::temporal::tgnn::{TgnModel, TgnConfig, TgnnGraph, TgnnEdge};
///
/// let config = TgnConfig::default();
/// let mut model = TgnModel::new(&config, 5, 0).expect("model");
///
/// let mut graph = TgnnGraph::with_zero_features(5, config.node_feat_dim);
/// graph.add_edge(TgnnEdge::no_feat(0, 1, 1.0));
/// graph.add_edge(TgnnEdge::no_feat(1, 2, 2.0));
///
/// model.process_events(&graph.edges).expect("process");
/// let embeddings = model.get_embeddings(&[0, 1, 2], 5.0, &graph).expect("embed");
/// assert_eq!(embeddings.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct TgnModel {
    /// Per-node memory
    pub memory: TgnMemory,
    /// Message computation module
    msg_module: TgnMessageModule,
    /// GRU memory updater
    updater: TgnMemoryUpdater,
    /// Temporal attention embedding
    embedding: TgnEmbedding,
    /// Configuration
    pub config: TgnConfig,
}

impl TgnModel {
    /// Create a new TGN model.
    ///
    /// `n_nodes`: number of nodes in the graph.
    /// `edge_feat_dim`: edge feature dimension (0 if no edge features).
    pub fn new(config: &TgnConfig, n_nodes: usize, edge_feat_dim: usize) -> Result<Self> {
        let memory = TgnMemory::new(n_nodes, config.memory_dim);
        let msg_module = TgnMessageModule::new(
            config.memory_dim,
            edge_feat_dim,
            config.time_dim,
            config.message_dim,
            77777,
        )?;
        let updater = TgnMemoryUpdater::new(config.memory_dim, config.message_dim, 88888);
        let embedding =
            TgnEmbedding::new(config.memory_dim, config.time_dim, config.embedding_dim, 99999)?;

        Ok(TgnModel {
            memory,
            msg_module,
            updater,
            embedding,
            config: config.clone(),
        })
    }

    /// Process a sequence of temporal events, updating node memories in order.
    ///
    /// Events are sorted by timestamp before processing (guarantees causal order).
    pub fn process_events(&mut self, events: &[TgnnEdge]) -> Result<()> {
        // Sort events by timestamp to ensure causal ordering
        let mut sorted: Vec<&TgnnEdge> = events.iter().collect();
        sorted.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for edge in sorted {
            let src = edge.src;
            let dst = edge.dst;
            let t = edge.timestamp;

            if src >= self.memory.n_nodes() || dst >= self.memory.n_nodes() {
                return Err(GraphError::InvalidParameter {
                    param: "edge.src/dst".to_string(),
                    value: format!("({}, {})", src, dst),
                    expected: format!("< {}", self.memory.n_nodes()),
                    context: "TgnModel::process_events".to_string(),
                });
            }

            // Snapshot memories before update (both nodes see each other's old state)
            let s_src = self.memory.get(src).to_vec();
            let s_dst = self.memory.get(dst).to_vec();
            let last_src = self.memory.last_update[src];
            let last_dst = self.memory.last_update[dst];

            // Compute messages
            let delta_t_src = (t - last_src).max(0.0);
            let delta_t_dst = (t - last_dst).max(0.0);

            let msg_src = self.msg_module.compute(&s_src, &s_dst, &edge.features, delta_t_src);
            let msg_dst = self.msg_module.compute(&s_dst, &s_src, &edge.features, delta_t_dst);

            // Update memories with GRU
            let new_s_src = self.updater.step(&s_src, &msg_src);
            let new_s_dst = self.updater.step(&s_dst, &msg_dst);

            self.memory.set(src, new_s_src, t)?;
            self.memory.set(dst, new_s_dst, t)?;
        }
        Ok(())
    }

    /// Compute embeddings for the given node indices at time `t`.
    ///
    /// Uses current memory states and temporal neighborhood from `graph`.
    pub fn get_embeddings(
        &self,
        nodes: &[usize],
        t: f64,
        graph: &TgnnGraph,
    ) -> Result<Vec<Vec<f64>>> {
        let mut result = Vec::with_capacity(nodes.len());
        for &node in nodes {
            if node >= self.memory.n_nodes() {
                return Err(GraphError::InvalidParameter {
                    param: "node".to_string(),
                    value: format!("{}", node),
                    expected: format!("< {}", self.memory.n_nodes()),
                    context: "TgnModel::get_embeddings".to_string(),
                });
            }
            let s_self = self.memory.get(node);
            let nbr_tuples = graph.neighbors_before(node, t);

            // Build (memory, timestamp) for each neighbor
            let neighbors: Vec<(&[f64], f64)> = nbr_tuples
                .iter()
                .filter_map(|(j, t_edge, _)| {
                    if *j < self.memory.n_nodes() {
                        Some((self.memory.get(*j), *t_edge))
                    } else {
                        None
                    }
                })
                .collect();

            let emb = self.embedding.embed_node(s_self, &neighbors, t);
            result.push(emb);
        }
        Ok(result)
    }

    /// Compute `TemporalPrediction` records for given nodes.
    pub fn predict(
        &self,
        nodes: &[usize],
        t: f64,
        graph: &TgnnGraph,
    ) -> Result<Vec<TemporalPrediction>> {
        let embeddings = self.get_embeddings(nodes, t, graph)?;
        Ok(nodes
            .iter()
            .zip(embeddings.into_iter())
            .map(|(&node, emb)| TemporalPrediction::new(node, emb, t))
            .collect())
    }

    /// Reset all node memories to zero.
    pub fn reset_memory(&mut self) {
        self.memory.reset();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::{TgnnEdge, TgnnGraph, TgnConfig};

    fn default_model(n_nodes: usize) -> TgnModel {
        let config = TgnConfig {
            memory_dim: 8,
            message_dim: 8,
            time_dim: 8,
            node_feat_dim: 4,
            embedding_dim: 8,
        };
        TgnModel::new(&config, n_nodes, 0).expect("model creation")
    }

    fn simple_graph() -> TgnnGraph {
        let mut g = TgnnGraph::with_zero_features(5, 4);
        g.add_edge(TgnnEdge::no_feat(0, 1, 1.0));
        g.add_edge(TgnnEdge::no_feat(1, 2, 2.0));
        g.add_edge(TgnnEdge::no_feat(2, 3, 3.0));
        g
    }

    #[test]
    fn test_tgn_memory_initialized_zeros() {
        let model = default_model(4);
        for i in 0..4 {
            let mem = model.memory.get(i);
            assert!(!mem.is_empty());
            let norm: f64 = mem.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm < 1e-12, "initial memory must be zero, got norm={}", norm);
        }
    }

    #[test]
    fn test_tgn_memory_updates_on_event() {
        let mut model = default_model(4);
        let events = vec![TgnnEdge::no_feat(0, 1, 1.0)];

        // Memories before processing
        let mem0_before: Vec<f64> = model.memory.get(0).to_vec();

        model.process_events(&events).expect("process events");

        let mem0_after: Vec<f64> = model.memory.get(0).to_vec();

        // Memory of node 0 must have changed
        let diff: f64 = mem0_before
            .iter()
            .zip(mem0_after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-10, "memory must change after event, diff={}", diff);
    }

    #[test]
    fn test_tgn_embeddings_shape() {
        let mut model = default_model(5);
        let graph = simple_graph();
        model.process_events(&graph.edges).expect("process");
        let embeddings = model
            .get_embeddings(&[0, 1, 2, 3, 4], 10.0, &graph)
            .expect("embeddings");
        assert_eq!(embeddings.len(), 5);
        for emb in &embeddings {
            assert_eq!(emb.len(), 8, "embedding dim must match config");
        }
    }

    #[test]
    fn test_tgn_with_no_edges() {
        // With no edges processed, all node memories stay at zero.
        // However embeddings are still non-zero because the time encoding φ(0)
        // has cos(ω_i·0)=1 components (non-zero), which get projected by W_v.
        // This is correct behaviour per the TGAT/TGN design.
        let model = default_model(3);
        let empty_graph = TgnnGraph::with_zero_features(3, 4);
        let embeddings = model
            .get_embeddings(&[0, 1, 2], 5.0, &empty_graph)
            .expect("embeddings");
        assert_eq!(embeddings.len(), 3, "must produce one embedding per queried node");
        // Each embedding must have the correct dimension
        for emb in &embeddings {
            assert_eq!(emb.len(), 8, "embedding dim must match config.embedding_dim");
        }
        // All nodes are symmetric (same zero memory, no neighbors at any t),
        // so all embeddings must be identical
        let ref_emb = &embeddings[0];
        for emb in &embeddings[1..] {
            let diff: f64 = emb.iter().zip(ref_emb.iter()).map(|(a, b)| (a - b).abs()).sum();
            assert!(
                diff < 1e-10,
                "all nodes with no events must have identical embeddings, diff={}",
                diff
            );
        }
    }

    #[test]
    fn test_tgn_gru_gate_range() {
        let updater = TgnMemoryUpdater::new(8, 8, 42);
        let h = vec![0.5f64; 8];
        let x = vec![-0.5f64; 8];
        let new_h = updater.step(&h, &x);

        // GRU output is a blend: (1-z)*h + z*tanh(...), all terms bounded
        // tanh output is in (-1,1), sigmoid in (0,1), so h_new in (-1,1)
        for &v in &new_h {
            assert!(
                v > -1.0 - 1e-6 && v < 1.0 + 1e-6,
                "GRU output must be in (-1, 1), got {}",
                v
            );
        }
    }

    #[test]
    fn test_tgn_message_dimension() {
        let memory_dim = 8;
        let edge_feat_dim = 4;
        let time_dim = 8;
        let message_dim = 16;

        let msg_module =
            TgnMessageModule::new(memory_dim, edge_feat_dim, time_dim, message_dim, 42)
                .expect("msg module");

        let s_src = vec![0.1f64; memory_dim];
        let s_dst = vec![0.2f64; memory_dim];
        let edge_feat = vec![0.5f64; edge_feat_dim];
        let delta_t = 1.5_f64;

        let msg = msg_module.compute(&s_src, &s_dst, &edge_feat, delta_t);
        assert_eq!(msg.len(), message_dim, "message must have message_dim elements");
    }

    #[test]
    fn test_tgn_event_ordering() {
        // Events processed out of order should produce same result as in-order
        let mut model_ordered = default_model(4);
        let events_ordered = vec![
            TgnnEdge::no_feat(0, 1, 1.0),
            TgnnEdge::no_feat(1, 2, 2.0),
            TgnnEdge::no_feat(2, 3, 3.0),
        ];
        model_ordered.process_events(&events_ordered).expect("ordered process");

        let mut model_shuffled = default_model(4);
        let events_shuffled = vec![
            TgnnEdge::no_feat(2, 3, 3.0),
            TgnnEdge::no_feat(0, 1, 1.0),
            TgnnEdge::no_feat(1, 2, 2.0),
        ];
        model_shuffled.process_events(&events_shuffled).expect("shuffled process");

        // Both models must have the same memory states since process_events sorts by time
        for i in 0..4 {
            let m1 = model_ordered.memory.get(i);
            let m2 = model_shuffled.memory.get(i);
            let diff: f64 = m1.iter().zip(m2.iter()).map(|(a, b)| (a - b).abs()).sum();
            assert!(
                diff < 1e-10,
                "event ordering: node {} memory mismatch, diff={}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_tgn_memory_gru_unit() {
        // Manual GRU step: with zero initial state and zero message,
        // the new state should also be zero (tanh(0)=0, σ(0)=0.5, so 0.5*0+0.5*0=0)
        let updater = TgnMemoryUpdater::new(4, 4, 0);
        // All weight matrices are Xavier init (near zero but not exactly zero)
        // With zero inputs, W·0 = 0, so z=σ(0)=0.5, r=σ(0)=0.5, h_tilde=tanh(0)=0
        // h_new = (1-0.5)*0 + 0.5*0 = 0
        let h_zero = vec![0.0f64; 4];
        let x_zero = vec![0.0f64; 4];
        let new_h = updater.step(&h_zero, &x_zero);

        let norm: f64 = new_h.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            norm < 1e-10,
            "GRU of zero inputs with zero h must give zero output, norm={}",
            norm
        );
    }

    #[test]
    fn test_tgn_process_out_of_bounds_node() {
        let mut model = default_model(3);
        let bad_events = vec![TgnnEdge::no_feat(0, 5, 1.0)]; // node 5 doesn't exist
        let result = model.process_events(&bad_events);
        assert!(result.is_err(), "out-of-bounds node must return error");
    }

    #[test]
    fn test_tgn_reset_memory() {
        let mut model = default_model(3);
        let events = vec![TgnnEdge::no_feat(0, 1, 1.0)];
        model.process_events(&events).expect("process");

        // Memory should have changed
        let before: Vec<f64> = model.memory.get(0).to_vec();
        let norm: f64 = before.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 1e-10, "memory should be non-zero after event");

        // Reset
        model.reset_memory();
        let after: Vec<f64> = model.memory.get(0).to_vec();
        let norm_after: f64 = after.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm_after < 1e-12, "memory should be zero after reset");
    }
}
