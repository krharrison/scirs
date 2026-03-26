//! Temporal Graph Neural Networks (TGN)
//!
//! Implements temporal graph neural network components inspired by
//! Rossi et al. (2020), "Temporal Graph Networks for Deep Learning on
//! Dynamic Graphs".
//!
//! Key components:
//! - **Temporal events**: continuous-time interactions between nodes
//! - **Time encoding**: learnable temporal representations (Time2Vec / sinusoidal)
//! - **Memory module**: GRU-based node memory updated on new events
//! - **Temporal attention**: attend to recent neighbors with time decay
//! - **Message function**: compute messages from temporal events

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt};

use crate::error::{GraphError, Result};

// ============================================================================
// Temporal Event
// ============================================================================

/// A temporal interaction event in a continuous-time dynamic graph.
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// Source node index
    pub source: usize,
    /// Target node index
    pub target: usize,
    /// Timestamp of the event
    pub timestamp: f64,
    /// Optional edge features associated with the event
    pub features: Option<Vec<f64>>,
}

impl TemporalEvent {
    /// Create a new temporal event.
    pub fn new(source: usize, target: usize, timestamp: f64) -> Self {
        TemporalEvent {
            source,
            target,
            timestamp,
            features: None,
        }
    }

    /// Create a temporal event with features.
    pub fn with_features(source: usize, target: usize, timestamp: f64, features: Vec<f64>) -> Self {
        TemporalEvent {
            source,
            target,
            timestamp,
            features: Some(features),
        }
    }
}

// ============================================================================
// Time Encoding
// ============================================================================

/// Type of time encoding to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeEncodingType {
    /// Sinusoidal encoding similar to positional encoding in Transformers
    Sinusoidal,
    /// Learnable Time2Vec encoding
    Time2Vec,
}

/// Temporal encoding module that maps scalar timestamps to vector representations.
///
/// Supports two encoding types:
/// - **Sinusoidal**: fixed frequencies at different scales
/// - **Time2Vec**: learnable linear + periodic components
#[derive(Debug, Clone)]
pub struct TimeEncoding {
    /// Encoding type
    pub encoding_type: TimeEncodingType,
    /// Output dimension for time encoding
    pub time_dim: usize,
    /// Learnable frequency parameters for Time2Vec: `[time_dim]`
    pub omega: Array1<f64>,
    /// Learnable phase parameters for Time2Vec: `[time_dim]`
    pub phi: Array1<f64>,
    /// Linear component weight for Time2Vec
    pub linear_weight: f64,
    /// Linear component bias for Time2Vec
    pub linear_bias: f64,
}

impl TimeEncoding {
    /// Create a new time encoding module.
    ///
    /// # Arguments
    /// * `time_dim` - Output dimension for temporal encoding
    /// * `encoding_type` - Type of encoding (sinusoidal or Time2Vec)
    pub fn new(time_dim: usize, encoding_type: TimeEncodingType) -> Self {
        let mut rng = scirs2_core::random::rng();

        let omega = match &encoding_type {
            TimeEncodingType::Sinusoidal => {
                // Fixed geometric frequencies
                Array1::from_iter(
                    (0..time_dim)
                        .map(|i| 1.0 / 10000.0_f64.powf(2.0 * (i / 2) as f64 / time_dim as f64)),
                )
            }
            TimeEncodingType::Time2Vec => {
                // Learnable frequencies initialized around 1.0
                Array1::from_iter((0..time_dim).map(|_| rng.random::<f64>() * 2.0))
            }
        };

        let phi =
            Array1::from_iter((0..time_dim).map(|_| rng.random::<f64>() * std::f64::consts::TAU));

        TimeEncoding {
            encoding_type,
            time_dim,
            omega,
            phi,
            linear_weight: rng.random::<f64>() * 0.1,
            linear_bias: 0.0,
        }
    }

    /// Encode a single timestamp.
    ///
    /// # Arguments
    /// * `t` - Scalar timestamp
    ///
    /// # Returns
    /// Time encoding vector of length `time_dim`
    pub fn encode(&self, t: f64) -> Array1<f64> {
        let mut encoding = Array1::zeros(self.time_dim);

        match self.encoding_type {
            TimeEncodingType::Sinusoidal => {
                for i in 0..self.time_dim {
                    let angle = t * self.omega[i];
                    if i % 2 == 0 {
                        encoding[i] = angle.sin();
                    } else {
                        encoding[i] = angle.cos();
                    }
                }
            }
            TimeEncodingType::Time2Vec => {
                // First component is linear
                if self.time_dim > 0 {
                    encoding[0] = self.linear_weight * t + self.linear_bias;
                }
                // Remaining components are periodic
                for i in 1..self.time_dim {
                    encoding[i] = (self.omega[i] * t + self.phi[i]).sin();
                }
            }
        }

        encoding
    }

    /// Encode multiple timestamps.
    ///
    /// # Arguments
    /// * `timestamps` - Slice of timestamps
    ///
    /// # Returns
    /// Matrix `[len, time_dim]` of time encodings
    pub fn encode_batch(&self, timestamps: &[f64]) -> Array2<f64> {
        let n = timestamps.len();
        let mut result = Array2::zeros((n, self.time_dim));
        for (i, &t) in timestamps.iter().enumerate() {
            let enc = self.encode(t);
            for j in 0..self.time_dim {
                result[[i, j]] = enc[j];
            }
        }
        result
    }
}

// ============================================================================
// Memory Module
// ============================================================================

/// Method for updating node memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryUpdateMethod {
    /// GRU-based memory update
    Gru,
    /// Simple concatenation + projection
    Mlp,
}

/// TGN-style memory module that maintains compressed node interaction history.
///
/// Each node has a memory vector that is updated when new events arrive.
/// The memory captures temporal patterns and interaction history.
#[derive(Debug, Clone)]
pub struct MemoryModule {
    /// Node memory states: `[n_nodes, memory_dim]`
    pub memory: Array2<f64>,
    /// Last update timestamp per node
    pub last_update: Vec<f64>,
    /// Memory dimension
    pub memory_dim: usize,
    /// Time encoding dimension
    pub time_dim: usize,
    /// Number of nodes
    pub n_nodes: usize,
    /// Memory update method
    pub update_method: MemoryUpdateMethod,
    /// Time encoding module
    pub time_encoding: TimeEncoding,
    /// Message dimension (memory_dim + memory_dim + time_dim + optional features)
    pub message_dim: usize,

    // GRU parameters (when update_method == Gru)
    /// GRU update gate: W_z `[message_dim, memory_dim]`
    gru_wz: Array2<f64>,
    /// GRU update gate: U_z `[memory_dim, memory_dim]`
    gru_uz: Array2<f64>,
    /// GRU reset gate: W_r `[message_dim, memory_dim]`
    gru_wr: Array2<f64>,
    /// GRU reset gate: U_r `[memory_dim, memory_dim]`
    gru_ur: Array2<f64>,
    /// GRU candidate: W_h `[message_dim, memory_dim]`
    gru_wh: Array2<f64>,
    /// GRU candidate: U_h `[memory_dim, memory_dim]`
    gru_uh: Array2<f64>,
    /// GRU biases
    gru_bz: Array1<f64>,
    /// GRU reset bias
    gru_br: Array1<f64>,
    /// GRU candidate bias
    gru_bh: Array1<f64>,

    // MLP parameters (when update_method == Mlp)
    /// MLP projection: `[message_dim + memory_dim, memory_dim]`
    mlp_w: Array2<f64>,
    /// MLP bias
    mlp_b: Array1<f64>,
}

impl MemoryModule {
    /// Create a new memory module.
    ///
    /// # Arguments
    /// * `n_nodes` - Number of nodes in the graph
    /// * `memory_dim` - Dimension of node memory vectors
    /// * `time_dim` - Dimension for time encoding
    /// * `update_method` - Memory update method (GRU or MLP)
    pub fn new(
        n_nodes: usize,
        memory_dim: usize,
        time_dim: usize,
        update_method: MemoryUpdateMethod,
    ) -> Self {
        let mut rng = scirs2_core::random::rng();
        let time_encoding = TimeEncoding::new(time_dim, TimeEncodingType::Time2Vec);

        // message = concat(source_memory, target_memory, time_encoding)
        let message_dim = memory_dim + memory_dim + time_dim;

        let scale_gru = (6.0_f64 / (message_dim + memory_dim) as f64).sqrt();
        let scale_u = (6.0_f64 / (2 * memory_dim) as f64).sqrt();
        let scale_mlp = (6.0_f64 / (message_dim + 2 * memory_dim) as f64).sqrt();

        let mut init = |r: usize, c: usize, s: f64| -> Array2<f64> {
            Array2::from_shape_fn((r, c), |_| (rng.random::<f64>() * 2.0 - 1.0) * s)
        };

        MemoryModule {
            memory: Array2::zeros((n_nodes, memory_dim)),
            last_update: vec![0.0; n_nodes],
            memory_dim,
            time_dim,
            n_nodes,
            update_method,
            time_encoding,
            message_dim,
            gru_wz: init(message_dim, memory_dim, scale_gru),
            gru_uz: init(memory_dim, memory_dim, scale_u),
            gru_wr: init(message_dim, memory_dim, scale_gru),
            gru_ur: init(memory_dim, memory_dim, scale_u),
            gru_wh: init(message_dim, memory_dim, scale_gru),
            gru_uh: init(memory_dim, memory_dim, scale_u),
            gru_bz: Array1::zeros(memory_dim),
            gru_br: Array1::zeros(memory_dim),
            gru_bh: Array1::zeros(memory_dim),
            mlp_w: init(message_dim + memory_dim, memory_dim, scale_mlp),
            mlp_b: Array1::zeros(memory_dim),
        }
    }

    /// Compute message from a temporal event.
    ///
    /// Message = concat(source_memory, target_memory, time_encoding(delta_t))
    fn compute_message(&self, event: &TemporalEvent) -> Vec<f64> {
        let src = event.source;
        let tgt = event.target;
        let delta_t = event.timestamp - self.last_update[src].max(self.last_update[tgt]);
        let time_enc = self.time_encoding.encode(delta_t);

        let mut msg = Vec::with_capacity(self.message_dim);

        // Source memory
        for j in 0..self.memory_dim {
            msg.push(if src < self.n_nodes {
                self.memory[[src, j]]
            } else {
                0.0
            });
        }

        // Target memory
        for j in 0..self.memory_dim {
            msg.push(if tgt < self.n_nodes {
                self.memory[[tgt, j]]
            } else {
                0.0
            });
        }

        // Time encoding
        for j in 0..self.time_dim {
            msg.push(time_enc[j]);
        }

        msg
    }

    /// GRU update step.
    ///
    /// ```text
    /// z = sigmoid(W_z * msg + U_z * h + b_z)
    /// r = sigmoid(W_r * msg + U_r * h + b_r)
    /// h_tilde = tanh(W_h * msg + U_h * (r * h) + b_h)
    /// h_new = (1 - z) * h + z * h_tilde
    /// ```
    fn gru_update(&self, memory: &[f64], message: &[f64]) -> Vec<f64> {
        let d = self.memory_dim;
        let m = self.message_dim;

        // Compute z (update gate)
        let mut z = vec![0.0f64; d];
        for j in 0..d {
            let mut s = self.gru_bz[j];
            for k in 0..m {
                s += message[k] * self.gru_wz[[k, j]];
            }
            for k in 0..d {
                s += memory[k] * self.gru_uz[[k, j]];
            }
            z[j] = sigmoid(s);
        }

        // Compute r (reset gate)
        let mut r = vec![0.0f64; d];
        for j in 0..d {
            let mut s = self.gru_br[j];
            for k in 0..m {
                s += message[k] * self.gru_wr[[k, j]];
            }
            for k in 0..d {
                s += memory[k] * self.gru_ur[[k, j]];
            }
            r[j] = sigmoid(s);
        }

        // Compute h_tilde (candidate)
        let mut h_tilde = vec![0.0f64; d];
        for j in 0..d {
            let mut s = self.gru_bh[j];
            for k in 0..m {
                s += message[k] * self.gru_wh[[k, j]];
            }
            for k in 0..d {
                s += (r[k] * memory[k]) * self.gru_uh[[k, j]];
            }
            h_tilde[j] = s.tanh();
        }

        // h_new = (1 - z) * h + z * h_tilde
        let mut h_new = vec![0.0f64; d];
        for j in 0..d {
            h_new[j] = (1.0 - z[j]) * memory[j] + z[j] * h_tilde[j];
        }

        h_new
    }

    /// MLP-based memory update.
    fn mlp_update(&self, memory: &[f64], message: &[f64]) -> Vec<f64> {
        let d = self.memory_dim;
        let total_in = self.message_dim + d;

        // Concatenate message + memory
        let mut input = Vec::with_capacity(total_in);
        input.extend_from_slice(message);
        input.extend_from_slice(memory);

        // Linear + tanh
        let mut out = vec![0.0f64; d];
        for j in 0..d {
            let mut s = self.mlp_b[j];
            for k in 0..total_in {
                s += input[k] * self.mlp_w[[k, j]];
            }
            out[j] = s.tanh();
        }

        out
    }

    /// Process a single temporal event and update node memories.
    ///
    /// Updates the memory of both source and target nodes involved in the event.
    ///
    /// # Arguments
    /// * `event` - The temporal interaction event
    pub fn process_event(&mut self, event: &TemporalEvent) -> Result<()> {
        if event.source >= self.n_nodes || event.target >= self.n_nodes {
            return Err(GraphError::InvalidParameter {
                param: "event".to_string(),
                value: format!("source={}, target={}", event.source, event.target),
                expected: format!("indices < {}", self.n_nodes),
                context: "MemoryModule::process_event".to_string(),
            });
        }

        let message = self.compute_message(event);

        // Update source memory
        let src_memory: Vec<f64> = (0..self.memory_dim)
            .map(|j| self.memory[[event.source, j]])
            .collect();
        let new_src = match self.update_method {
            MemoryUpdateMethod::Gru => self.gru_update(&src_memory, &message),
            MemoryUpdateMethod::Mlp => self.mlp_update(&src_memory, &message),
        };

        // Update target memory
        let tgt_memory: Vec<f64> = (0..self.memory_dim)
            .map(|j| self.memory[[event.target, j]])
            .collect();
        let new_tgt = match self.update_method {
            MemoryUpdateMethod::Gru => self.gru_update(&tgt_memory, &message),
            MemoryUpdateMethod::Mlp => self.mlp_update(&tgt_memory, &message),
        };

        // Write back
        for j in 0..self.memory_dim {
            self.memory[[event.source, j]] = new_src[j];
            self.memory[[event.target, j]] = new_tgt[j];
        }

        self.last_update[event.source] = event.timestamp;
        self.last_update[event.target] = event.timestamp;

        Ok(())
    }

    /// Process a batch of temporal events in chronological order.
    ///
    /// Events should be sorted by timestamp (ascending).
    pub fn process_events(&mut self, events: &[TemporalEvent]) -> Result<()> {
        for event in events {
            self.process_event(event)?;
        }
        Ok(())
    }

    /// Get the current memory state for all nodes.
    pub fn get_memory(&self) -> &Array2<f64> {
        &self.memory
    }

    /// Reset all node memories to zero.
    pub fn reset(&mut self) {
        self.memory.fill(0.0);
        self.last_update.fill(0.0);
    }
}

/// Sigmoid activation function.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// Temporal Attention
// ============================================================================

/// Temporal attention mechanism that attends to recent neighbors
/// with time-decay weighting.
///
/// For each node, computes attention over its recent interaction partners,
/// incorporating temporal information through time encodings.
#[derive(Debug, Clone)]
pub struct TemporalAttention {
    /// Query projection: `[memory_dim, hidden_dim]`
    pub w_q: Array2<f64>,
    /// Key projection: `[memory_dim + time_dim, hidden_dim]`
    pub w_k: Array2<f64>,
    /// Value projection: `[memory_dim + time_dim, hidden_dim]`
    pub w_v: Array2<f64>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Time encoding module
    pub time_encoding: TimeEncoding,
    /// Memory dimension
    pub memory_dim: usize,
    /// Time encoding dimension
    pub time_dim: usize,
}

impl TemporalAttention {
    /// Create a new temporal attention module.
    ///
    /// # Arguments
    /// * `memory_dim` - Dimension of node memory
    /// * `time_dim` - Dimension of time encoding
    /// * `num_heads` - Number of attention heads
    pub fn new(memory_dim: usize, time_dim: usize, num_heads: usize) -> Result<Self> {
        let hidden_dim = memory_dim;
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(GraphError::InvalidParameter {
                param: "memory_dim".to_string(),
                value: format!("{memory_dim}"),
                expected: format!("divisible by num_heads={num_heads}"),
                context: "TemporalAttention::new".to_string(),
            });
        }

        let head_dim = hidden_dim / num_heads;
        let mut rng = scirs2_core::random::rng();
        let scale_q = (6.0_f64 / (memory_dim + hidden_dim) as f64).sqrt();
        let scale_kv = (6.0_f64 / (memory_dim + time_dim + hidden_dim) as f64).sqrt();

        let w_q = Array2::from_shape_fn((memory_dim, hidden_dim), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * scale_q
        });
        let w_k = Array2::from_shape_fn((memory_dim + time_dim, hidden_dim), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * scale_kv
        });
        let w_v = Array2::from_shape_fn((memory_dim + time_dim, hidden_dim), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) * scale_kv
        });

        let time_encoding = TimeEncoding::new(time_dim, TimeEncodingType::Sinusoidal);

        Ok(TemporalAttention {
            w_q,
            w_k,
            w_v,
            num_heads,
            hidden_dim,
            head_dim,
            time_encoding,
            memory_dim,
            time_dim,
        })
    }

    /// Compute temporal attention for a single node given its neighbors and event times.
    ///
    /// # Arguments
    /// * `query_memory` - Memory of the query node `[memory_dim]`
    /// * `neighbor_memories` - Memory vectors of neighbor nodes `[num_neighbors, memory_dim]`
    /// * `time_deltas` - Time differences for each neighbor interaction `[num_neighbors]`
    ///
    /// # Returns
    /// Aggregated representation `[hidden_dim]`
    pub fn forward(
        &self,
        query_memory: &Array1<f64>,
        neighbor_memories: &Array2<f64>,
        time_deltas: &[f64],
    ) -> Result<Array1<f64>> {
        let num_neighbors = neighbor_memories.dim().0;
        if num_neighbors == 0 {
            return Ok(Array1::zeros(self.hidden_dim));
        }
        if time_deltas.len() != num_neighbors {
            return Err(GraphError::InvalidParameter {
                param: "time_deltas".to_string(),
                value: format!("len={}", time_deltas.len()),
                expected: format!("len={num_neighbors}"),
                context: "TemporalAttention::forward".to_string(),
            });
        }

        let h = self.num_heads;
        let dk = self.head_dim;
        let scale = 1.0 / (dk as f64).sqrt();

        // Query: W_q * query_memory -> [hidden_dim]
        let mut q = vec![0.0f64; self.hidden_dim];
        for j in 0..self.hidden_dim {
            for m in 0..self.memory_dim {
                q[j] += query_memory[m] * self.w_q[[m, j]];
            }
        }

        // For each neighbor, compute key and value
        let kv_in_dim = self.memory_dim + self.time_dim;
        let mut keys = vec![vec![0.0f64; self.hidden_dim]; num_neighbors];
        let mut values = vec![vec![0.0f64; self.hidden_dim]; num_neighbors];

        for nb in 0..num_neighbors {
            // Concatenate neighbor memory with time encoding
            let time_enc = self.time_encoding.encode(time_deltas[nb]);
            let mut kv_input = Vec::with_capacity(kv_in_dim);
            for m in 0..self.memory_dim {
                kv_input.push(neighbor_memories[[nb, m]]);
            }
            for m in 0..self.time_dim {
                kv_input.push(time_enc[m]);
            }

            for j in 0..self.hidden_dim {
                let mut sk = 0.0;
                let mut sv = 0.0;
                for m in 0..kv_in_dim {
                    sk += kv_input[m] * self.w_k[[m, j]];
                    sv += kv_input[m] * self.w_v[[m, j]];
                }
                keys[nb][j] = sk;
                values[nb][j] = sv;
            }
        }

        // Multi-head attention
        let mut output = vec![0.0f64; self.hidden_dim];

        for head in 0..h {
            let offset = head * dk;

            // Scores
            let mut scores = vec![0.0f64; num_neighbors];
            for nb in 0..num_neighbors {
                let mut dot = 0.0;
                for m in 0..dk {
                    dot += q[offset + m] * keys[nb][offset + m];
                }
                scores[nb] = dot * scale;
            }

            // Softmax
            let alphas = softmax_slice(&scores);

            // Aggregate
            for nb in 0..num_neighbors {
                for m in 0..dk {
                    output[offset + m] += alphas[nb] * values[nb][offset + m];
                }
            }
        }

        Ok(Array1::from_vec(output))
    }
}

/// Numerically-stable softmax.
fn softmax_slice(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let max_val = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|x| (x - max_val).exp()).collect();
    let sum = exps.iter().sum::<f64>().max(1e-12);
    exps.iter().map(|e| e / sum).collect()
}

// ============================================================================
// TGN Configuration
// ============================================================================

/// Configuration for the Temporal GNN model.
#[derive(Debug, Clone)]
pub struct TemporalGnnConfig {
    /// Number of nodes
    pub n_nodes: usize,
    /// Memory dimension
    pub memory_dim: usize,
    /// Time encoding dimension
    pub time_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Memory update method
    pub update_method: MemoryUpdateMethod,
    /// Time encoding type
    pub time_encoding_type: TimeEncodingType,
}

impl Default for TemporalGnnConfig {
    fn default() -> Self {
        TemporalGnnConfig {
            n_nodes: 100,
            memory_dim: 64,
            time_dim: 16,
            num_heads: 4,
            update_method: MemoryUpdateMethod::Gru,
            time_encoding_type: TimeEncodingType::Time2Vec,
        }
    }
}

// ============================================================================
// Temporal GNN Model
// ============================================================================

/// Full Temporal Graph Neural Network model.
///
/// Combines a TGN-style memory module with temporal attention for
/// processing continuous-time dynamic graphs.
#[derive(Debug, Clone)]
pub struct TemporalGnnModel {
    /// Memory module
    pub memory_module: MemoryModule,
    /// Temporal attention
    pub temporal_attention: TemporalAttention,
    /// Configuration
    pub config: TemporalGnnConfig,
    /// Event history for neighbor lookup
    event_history: Vec<TemporalEvent>,
}

impl TemporalGnnModel {
    /// Create a new Temporal GNN model.
    pub fn new(config: TemporalGnnConfig) -> Result<Self> {
        let memory_module = MemoryModule::new(
            config.n_nodes,
            config.memory_dim,
            config.time_dim,
            config.update_method.clone(),
        );
        let temporal_attention =
            TemporalAttention::new(config.memory_dim, config.time_dim, config.num_heads)?;

        Ok(TemporalGnnModel {
            memory_module,
            temporal_attention,
            config,
            event_history: Vec::new(),
        })
    }

    /// Process a batch of events and update model state.
    ///
    /// Events should be in chronological order.
    pub fn process_events(&mut self, events: &[TemporalEvent]) -> Result<()> {
        self.memory_module.process_events(events)?;
        self.event_history.extend(events.iter().cloned());
        Ok(())
    }

    /// Get node embedding at a given time by aggregating over recent neighbors.
    ///
    /// # Arguments
    /// * `node` - Node index
    /// * `current_time` - Current timestamp for computing time deltas
    /// * `max_neighbors` - Maximum number of recent neighbors to attend to
    ///
    /// # Returns
    /// Node embedding `[memory_dim]`
    pub fn get_node_embedding(
        &self,
        node: usize,
        current_time: f64,
        max_neighbors: usize,
    ) -> Result<Array1<f64>> {
        if node >= self.config.n_nodes {
            return Err(GraphError::InvalidParameter {
                param: "node".to_string(),
                value: format!("{node}"),
                expected: format!("< {}", self.config.n_nodes),
                context: "TemporalGnnModel::get_node_embedding".to_string(),
            });
        }

        // Find recent neighbors from event history
        let mut neighbor_events: Vec<(usize, f64)> = Vec::new();
        for event in self.event_history.iter().rev() {
            if neighbor_events.len() >= max_neighbors {
                break;
            }
            if event.source == node {
                neighbor_events.push((event.target, event.timestamp));
            } else if event.target == node {
                neighbor_events.push((event.source, event.timestamp));
            }
        }

        if neighbor_events.is_empty() {
            // Return raw memory if no neighbors found
            return Ok(Array1::from_iter(
                (0..self.config.memory_dim).map(|j| self.memory_module.memory[[node, j]]),
            ));
        }

        // Build neighbor memory matrix and time deltas
        let num_nb = neighbor_events.len();
        let mut nb_memories = Array2::zeros((num_nb, self.config.memory_dim));
        let mut time_deltas = vec![0.0f64; num_nb];

        for (idx, &(nb_node, nb_time)) in neighbor_events.iter().enumerate() {
            for j in 0..self.config.memory_dim {
                nb_memories[[idx, j]] = self.memory_module.memory[[nb_node, j]];
            }
            time_deltas[idx] = current_time - nb_time;
        }

        // Query memory
        let query = Array1::from_iter(
            (0..self.config.memory_dim).map(|j| self.memory_module.memory[[node, j]]),
        );

        self.temporal_attention
            .forward(&query, &nb_memories, &time_deltas)
    }

    /// Reset the model state (memory and event history).
    pub fn reset(&mut self) {
        self.memory_module.reset();
        self.event_history.clear();
    }

    /// Get the current memory state.
    pub fn get_memory(&self) -> &Array2<f64> {
        self.memory_module.get_memory()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_encoding_sinusoidal_varies_with_time() {
        let te = TimeEncoding::new(8, TimeEncodingType::Sinusoidal);
        let enc1 = te.encode(0.0);
        let enc2 = te.encode(1.0);
        let enc3 = te.encode(10.0);

        assert_eq!(enc1.len(), 8);
        assert_eq!(enc2.len(), 8);

        // Different timestamps should produce different encodings
        let diff_12: f64 = enc1
            .iter()
            .zip(enc2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let diff_13: f64 = enc1
            .iter()
            .zip(enc3.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff_12 > 1e-6, "encodings at t=0 and t=1 should differ");
        assert!(diff_13 > 1e-6, "encodings at t=0 and t=10 should differ");
    }

    #[test]
    fn test_time_encoding_time2vec() {
        let te = TimeEncoding::new(6, TimeEncodingType::Time2Vec);
        let enc = te.encode(5.0);
        assert_eq!(enc.len(), 6);
        for &v in enc.iter() {
            assert!(v.is_finite(), "Time2Vec encoding should be finite");
        }

        // First component should be linear
        let enc0 = te.encode(0.0);
        let enc10 = te.encode(10.0);
        // Linear component: w * t + b
        let expected_diff = te.linear_weight * 10.0;
        let actual_diff = enc10[0] - enc0[0];
        assert!(
            (actual_diff - expected_diff).abs() < 1e-10,
            "first component should be linear"
        );
    }

    #[test]
    fn test_memory_update_changes_state() {
        let mut mem = MemoryModule::new(5, 8, 4, MemoryUpdateMethod::Gru);

        // Initially all zeros
        let initial_norm: f64 = mem.memory.iter().map(|x| x * x).sum();
        assert!(initial_norm < 1e-12, "initial memory should be zero");

        // Process an event
        let event = TemporalEvent::new(0, 1, 1.0);
        mem.process_event(&event).expect("process event");

        // Memory should have changed for nodes 0 and 1
        let node0_norm: f64 = (0..8).map(|j| mem.memory[[0, j]].powi(2)).sum();
        let node1_norm: f64 = (0..8).map(|j| mem.memory[[1, j]].powi(2)).sum();

        assert!(
            node0_norm > 1e-12,
            "node 0 memory should be updated after event"
        );
        assert!(
            node1_norm > 1e-12,
            "node 1 memory should be updated after event"
        );

        // Node 2 should still be zero (not involved in event)
        let node2_norm: f64 = (0..8).map(|j| mem.memory[[2, j]].powi(2)).sum();
        assert!(node2_norm < 1e-12, "node 2 memory should remain zero");
    }

    #[test]
    fn test_memory_update_mlp() {
        let mut mem = MemoryModule::new(4, 6, 3, MemoryUpdateMethod::Mlp);
        let event = TemporalEvent::new(0, 1, 0.5);
        mem.process_event(&event).expect("process event MLP");

        let node0_norm: f64 = (0..6).map(|j| mem.memory[[0, j]].powi(2)).sum();
        assert!(node0_norm > 1e-12, "MLP update should modify memory");
    }

    #[test]
    fn test_temporal_attention_shape() {
        let ta = TemporalAttention::new(8, 4, 2).expect("temporal attention");
        let query = Array1::from_vec(vec![0.1; 8]);
        let neighbors = Array2::from_shape_fn((3, 8), |(i, j)| (i + j) as f64 * 0.05);
        let deltas = vec![1.0, 2.0, 3.0];

        let out = ta.forward(&query, &neighbors, &deltas).expect("forward");
        assert_eq!(out.len(), 8);
        for &v in out.iter() {
            assert!(v.is_finite(), "temporal attention output should be finite");
        }
    }

    #[test]
    fn test_temporal_attention_empty_neighbors() {
        let ta = TemporalAttention::new(8, 4, 2).expect("temporal attention");
        let query = Array1::from_vec(vec![0.1; 8]);
        let neighbors = Array2::zeros((0, 8));
        let deltas: Vec<f64> = Vec::new();

        let out = ta
            .forward(&query, &neighbors, &deltas)
            .expect("empty forward");
        assert_eq!(out.len(), 8);
        // Should be all zeros for empty neighbors
        let norm: f64 = out.iter().map(|x| x * x).sum();
        assert!(norm < 1e-12, "empty neighbor attention should return zeros");
    }

    #[test]
    fn test_temporal_gnn_model_full_pipeline() {
        let config = TemporalGnnConfig {
            n_nodes: 5,
            memory_dim: 8,
            time_dim: 4,
            num_heads: 2,
            update_method: MemoryUpdateMethod::Gru,
            time_encoding_type: TimeEncodingType::Time2Vec,
        };

        let mut model = TemporalGnnModel::new(config).expect("model");

        // Process events
        let events = vec![
            TemporalEvent::new(0, 1, 1.0),
            TemporalEvent::new(1, 2, 2.0),
            TemporalEvent::new(0, 2, 3.0),
            TemporalEvent::new(2, 3, 4.0),
        ];
        model.process_events(&events).expect("process events");

        // Get embeddings
        let emb0 = model.get_node_embedding(0, 5.0, 3).expect("embedding 0");
        let emb4 = model.get_node_embedding(4, 5.0, 3).expect("embedding 4");

        assert_eq!(emb0.len(), 8);
        assert_eq!(emb4.len(), 8);

        // Node 0 has interactions, node 4 does not - embeddings should differ
        let diff: f64 = emb0
            .iter()
            .zip(emb4.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        // Node 4 returns raw zero memory, node 0 has aggregated info
        assert!(
            emb0.iter().any(|&v| v.abs() > 1e-12),
            "active node should have non-zero embedding"
        );
    }

    #[test]
    fn test_memory_module_event_out_of_bounds() {
        let mut mem = MemoryModule::new(3, 4, 2, MemoryUpdateMethod::Gru);
        let event = TemporalEvent::new(0, 5, 1.0); // node 5 > n_nodes=3
        let result = mem.process_event(&event);
        assert!(result.is_err());
    }

    #[test]
    fn test_temporal_gnn_reset() {
        let config = TemporalGnnConfig {
            n_nodes: 3,
            memory_dim: 4,
            time_dim: 2,
            num_heads: 2,
            ..Default::default()
        };

        let mut model = TemporalGnnModel::new(config).expect("model");
        let event = TemporalEvent::new(0, 1, 1.0);
        model.process_events(&[event]).expect("process");

        // After reset, memory should be zero again
        model.reset();
        let mem_norm: f64 = model.get_memory().iter().map(|x| x * x).sum();
        assert!(mem_norm < 1e-12, "memory should be zero after reset");
    }

    #[test]
    fn test_time_encoding_batch() {
        let te = TimeEncoding::new(4, TimeEncodingType::Sinusoidal);
        let timestamps = vec![0.0, 1.0, 5.0, 10.0];
        let batch = te.encode_batch(&timestamps);
        assert_eq!(batch.dim(), (4, 4));

        // Each row should match individual encoding
        for (i, &t) in timestamps.iter().enumerate() {
            let single = te.encode(t);
            for j in 0..4 {
                assert!(
                    (batch[[i, j]] - single[j]).abs() < 1e-12,
                    "batch encoding should match single encoding"
                );
            }
        }
    }

    #[test]
    fn test_memory_timestamps_updated() {
        let mut mem = MemoryModule::new(3, 4, 2, MemoryUpdateMethod::Gru);

        assert!(mem.last_update[0] < 1e-12);
        assert!(mem.last_update[1] < 1e-12);

        let event = TemporalEvent::new(0, 1, 5.0);
        mem.process_event(&event).expect("process");

        assert!(
            (mem.last_update[0] - 5.0).abs() < 1e-12,
            "source timestamp should be updated"
        );
        assert!(
            (mem.last_update[1] - 5.0).abs() < 1e-12,
            "target timestamp should be updated"
        );
        assert!(
            mem.last_update[2] < 1e-12,
            "uninvolved node timestamp should remain 0"
        );
    }
}
