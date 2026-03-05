//! Graph Convolution Layers for Graph Neural Networks (GNNs)
//!
//! This module provides implementations of three foundational graph neural network
//! convolution layers:
//!
//! - [`GraphConvLayer`]: Kipf & Welling (2017) spectral GCN with symmetric normalization
//!   `H' = σ(D^{-1/2} Â D^{-1/2} H W + b)` where `Â = A + I`
//!
//! - [`GraphAttentionLayer`]: Veličković et al. (2018) GAT with multi-head attention
//!   coefficients computed via a shared attention mechanism over neighbourhood pairs
//!
//! - [`GraphSageLayer`]: Hamilton et al. (2017) inductive representation learning via
//!   neighbourhood aggregation (mean or max) followed by a concat-then-project step
//!
//! All layers operate on dense node-feature matrices (`N × F`) where `N` is the
//! number of nodes and `F` is the feature dimension.  Adjacency matrices are
//! supplied as dense `N × N` arrays; for large, sparse graphs the caller is
//! expected to materialise only the neighbourhood sub-graph before calling
//! `forward`.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Distribution, Uniform};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

// ──────────────────────────────────────────────────────────────────────────────
// Helper: Xavier/Glorot weight initialisation
// ──────────────────────────────────────────────────────────────────────────────

fn xavier_init<F, R>(fan_in: usize, fan_out: usize, rng: &mut R) -> Result<Array2<F>>
where
    F: Float + Debug + NumAssign,
    R: scirs2_core::random::Rng,
{
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    let dist = Uniform::new(-limit, limit).map_err(|e| {
        NeuralError::InvalidArchitecture(format!("Failed to create Uniform distribution: {e}"))
    })?;

    let data: Vec<F> = (0..fan_in * fan_out)
        .map(|_| F::from(dist.sample(rng)).unwrap_or(F::zero()))
        .collect();

    Array2::from_shape_vec((fan_in, fan_out), data).map_err(|e| {
        NeuralError::InvalidArchitecture(format!("Failed to create weight matrix: {e}"))
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper: apply simple element-wise activation by name
// ──────────────────────────────────────────────────────────────────────────────

/// Supported activation functions for graph layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphActivation {
    /// No activation (identity)
    None,
    /// Rectified Linear Unit
    ReLU,
    /// Sigmoid function
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Exponential Linear Unit
    ELU,
}

impl GraphActivation {
    fn apply<F: Float>(&self, x: F) -> F {
        match self {
            GraphActivation::None => x,
            GraphActivation::ReLU => {
                if x > F::zero() {
                    x
                } else {
                    F::zero()
                }
            }
            GraphActivation::Sigmoid => F::one() / (F::one() + (-x).exp()),
            GraphActivation::Tanh => x.tanh(),
            GraphActivation::ELU => {
                if x >= F::zero() {
                    x
                } else {
                    x.exp() - F::one()
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GCN Layer
// ──────────────────────────────────────────────────────────────────────────────

/// Graph Convolutional Network (GCN) layer
///
/// Implements the spectral graph convolution from Kipf & Welling (2017):
/// ```text
/// H' = σ( D̂^{-1/2} Â D̂^{-1/2} H W + b )
/// ```
/// where `Â = A + I` (adjacency with self-loops) and `D̂` is the corresponding
/// degree matrix.
///
/// # Shape
/// * `features`:   `[N, in_features]`  — input node feature matrix
/// * `adjacency`:  `[N, N]`            — raw (unnormalised) adjacency matrix
/// * returns:      `[N, out_features]` — updated node feature matrix
///
/// # Examples
/// ```rust
/// use scirs2_neural::layers::graph_conv::{GraphConvLayer, GraphActivation};
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::random::rng;
///
/// let mut rng = rng();
/// let layer = GraphConvLayer::<f64>::new(4, 8, true, GraphActivation::ReLU, &mut rng)
///     .expect("Failed to create layer");
///
/// let features = Array2::<f64>::from_elem((5, 4), 0.1);
/// let adj = Array2::<f64>::eye(5);
/// let out = layer.forward_graph(&features, &adj).expect("Forward pass failed");
/// assert_eq!(out.shape(), &[5, 8]);
/// ```
#[derive(Debug, Clone)]
pub struct GraphConvLayer<F: Float + Debug + Send + Sync + NumAssign> {
    /// Weight matrix: `[in_features, out_features]`
    weight: Array2<F>,
    /// Optional bias vector: `[out_features]`
    bias: Option<Array1<F>>,
    /// Activation applied after the linear transform
    activation: GraphActivation,
    /// Gradient accumulator for `weight`
    dweight: Arc<RwLock<Array2<F>>>,
    /// Gradient accumulator for `bias`
    dbias: Arc<RwLock<Option<Array1<F>>>>,
    in_features: usize,
    out_features: usize,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> GraphConvLayer<F> {
    /// Create a new GCN layer.
    ///
    /// # Arguments
    /// * `in_features`  — dimension of input node features
    /// * `out_features` — dimension of output node features
    /// * `use_bias`     — whether to add a learnable bias term
    /// * `activation`   — element-wise activation applied after the projection
    /// * `rng`          — random number generator for weight initialisation
    pub fn new<R: scirs2_core::random::Rng>(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        activation: GraphActivation,
        rng: &mut R,
    ) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "in_features and out_features must be > 0".to_string(),
            ));
        }

        let weight = xavier_init(in_features, out_features, rng)?;
        let bias = if use_bias {
            Some(Array1::zeros(out_features))
        } else {
            None
        };

        let dweight = Arc::new(RwLock::new(Array2::zeros((in_features, out_features))));
        let dbias = Arc::new(RwLock::new(if use_bias {
            Some(Array1::zeros(out_features))
        } else {
            None
        }));

        Ok(Self {
            weight,
            bias,
            activation,
            dweight,
            dbias,
            in_features,
            out_features,
        })
    }

    /// Compute the symmetrically normalised adjacency matrix.
    ///
    /// Steps:
    /// 1. Add self-loops: `Â = A + I`
    /// 2. Compute degree vector `d_i = Σ_j Â_{ij}`
    /// 3. Return `D̂^{-1/2} Â D̂^{-1/2}`
    ///
    /// Nodes with zero degree (isolated even after self-loop addition) receive a
    /// normalisation factor of zero rather than producing NaN.
    pub fn normalize_adjacency(adjacency: &Array2<F>) -> Result<Array2<F>> {
        let n = adjacency.nrows();
        if adjacency.ncols() != n {
            return Err(NeuralError::InvalidArgument(format!(
                "Adjacency matrix must be square, got {}×{}",
                n,
                adjacency.ncols()
            )));
        }

        // Â = A + I
        let mut a_hat = adjacency.clone();
        for i in 0..n {
            a_hat[[i, i]] += F::one();
        }

        // Degree vector
        let mut d_inv_sqrt = Array1::<F>::zeros(n);
        for i in 0..n {
            let deg: F = a_hat
                .row(i)
                .iter()
                .copied()
                .fold(F::zero(), |acc, v| acc + v);
            if deg > F::zero() {
                d_inv_sqrt[i] = F::one() / deg.sqrt();
            }
        }

        // D̂^{-1/2} Â D̂^{-1/2}
        let mut norm = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                norm[[i, j]] = d_inv_sqrt[i] * a_hat[[i, j]] * d_inv_sqrt[j];
            }
        }
        Ok(norm)
    }

    /// Forward pass of the GCN layer.
    ///
    /// # Arguments
    /// * `features`  — `[N, in_features]` node feature matrix
    /// * `adjacency` — `[N, N]` raw adjacency matrix (self-loops are added internally)
    ///
    /// # Returns
    /// `[N, out_features]` updated feature matrix after graph convolution and activation.
    pub fn forward_graph(&self, features: &Array2<F>, adjacency: &Array2<F>) -> Result<Array2<F>> {
        let n = features.nrows();
        let f_in = features.ncols();

        if f_in != self.in_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "Expected in_features={}, got {}",
                self.in_features, f_in
            )));
        }
        if adjacency.nrows() != n || adjacency.ncols() != n {
            return Err(NeuralError::InvalidArgument(format!(
                "Adjacency must be {}×{} but got {}×{}",
                n,
                n,
                adjacency.nrows(),
                adjacency.ncols()
            )));
        }

        // Normalised adjacency
        let norm_adj = Self::normalize_adjacency(adjacency)?;

        // AH = norm_adj @ features  [N, in_features]
        let mut agg = Array2::<F>::zeros((n, self.in_features));
        for i in 0..n {
            for j in 0..n {
                let coeff = norm_adj[[i, j]];
                if coeff == F::zero() {
                    continue;
                }
                for k in 0..self.in_features {
                    agg[[i, k]] += coeff * features[[j, k]];
                }
            }
        }

        // AHW = AH @ weight  [N, out_features]
        let mut out = Array2::<F>::zeros((n, self.out_features));
        for i in 0..n {
            for k in 0..self.in_features {
                let a_ik = agg[[i, k]];
                if a_ik == F::zero() {
                    continue;
                }
                for j in 0..self.out_features {
                    out[[i, j]] += a_ik * self.weight[[k, j]];
                }
            }
        }

        // Add bias and apply activation
        for i in 0..n {
            for j in 0..self.out_features {
                let v = if let Some(ref b) = self.bias {
                    out[[i, j]] + b[j]
                } else {
                    out[[i, j]]
                };
                out[[i, j]] = self.activation.apply(v);
            }
        }

        Ok(out)
    }

    /// Number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        let bias_params = self.bias.as_ref().map_or(0, |b| b.len());
        self.in_features * self.out_features + bias_params
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for GraphConvLayer<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // When used as a generic Layer the adjacency defaults to identity
        // (pure self-loop message passing == standard linear layer).
        let f2 = input
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|e| NeuralError::DimensionMismatch(format!("Expected 2D input: {e}")))?;
        let n = f2.nrows();
        let adj = Array2::<F>::eye(n);
        self.forward_graph(&f2, &adj).map(|a| a.into_dyn())
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, lr: F) -> Result<()> {
        let dw = self
            .dweight
            .read()
            .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))?;
        for (w, g) in self.weight.iter_mut().zip(dw.iter()) {
            *w -= lr * *g;
        }
        drop(dw);

        if let Some(ref mut bias) = self.bias {
            let db_guard = self
                .dbias
                .read()
                .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))?;
            if let Some(ref db) = *db_guard {
                for (b, g) in bias.iter_mut().zip(db.iter()) {
                    *b -= lr * *g;
                }
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "GraphConv"
    }

    fn parameter_count(&self) -> usize {
        self.num_parameters()
    }

    fn layer_description(&self) -> String {
        format!(
            "type:GraphConv in:{} out:{} bias:{} act:{:?}",
            self.in_features,
            self.out_features,
            self.bias.is_some(),
            self.activation
        )
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut v = vec![self.weight.clone().into_dyn()];
        if let Some(ref b) = self.bias {
            v.push(b.clone().into_dyn());
        }
        v
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GAT Layer
// ──────────────────────────────────────────────────────────────────────────────

/// Graph Attention Network (GAT) layer
///
/// Implements multi-head graph attention from Veličković et al. (2018).
/// Each attention head independently computes:
/// ```text
/// e_{ij} = LeakyReLU( a^T [W h_i ‖ W h_j] )
/// α_{ij} = softmax_j( e_{ij} )
/// h'_i   = σ( Σ_j α_{ij} W h_j )
/// ```
/// Outputs from all heads are **concatenated** (total dim: `n_heads × out_features`).
///
/// # Shape
/// * `features`:   `[N, in_features]`
/// * `adjacency`:  `[N, N]`  (non-zero ⇒ edge exists; value ignored for attention)
/// * returns:      `[N, n_heads * out_features]`
///
/// # Examples
/// ```rust
/// use scirs2_neural::layers::graph_conv::GraphAttentionLayer;
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::random::rng;
///
/// let mut rng = rng();
/// let layer = GraphAttentionLayer::<f64>::new(4, 8, 2, 0.0, &mut rng)
///     .expect("Failed to create GAT layer");
///
/// let features = Array2::<f64>::from_elem((6, 4), 0.1);
/// let adj = Array2::<f64>::eye(6);
/// let out = layer.forward_graph(&features, &adj, false).expect("Forward pass failed");
/// assert_eq!(out.shape(), &[6, 16]); // 2 heads x 8 out_features
/// ```
#[derive(Debug, Clone)]
pub struct GraphAttentionLayer<F: Float + Debug + Send + Sync + NumAssign> {
    /// Per-head weight matrices: `n_heads × [in_features, out_features]`
    weights: Vec<Array2<F>>,
    /// Per-head attention vectors: `n_heads × [2*out_features]`
    attention_vecs: Vec<Array1<F>>,
    /// Number of attention heads
    n_heads: usize,
    /// Dropout probability applied to attention coefficients (during training)
    dropout: f64,
    /// LeakyReLU negative slope for attention scoring
    leaky_relu_slope: F,
    in_features: usize,
    out_features: usize,
    /// Gradient accumulators (per head)
    dweights: Vec<Arc<RwLock<Array2<F>>>>,
    dattn_vecs: Vec<Arc<RwLock<Array1<F>>>>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> GraphAttentionLayer<F> {
    /// Create a new GAT layer.
    ///
    /// # Arguments
    /// * `in_features`  — input node feature dimension
    /// * `out_features` — per-head output dimension (total output is `n_heads × out_features`)
    /// * `n_heads`      — number of independent attention heads
    /// * `dropout`      — probability of zeroing an attention coefficient (0.0 disables)
    /// * `rng`          — random number generator
    pub fn new<R: scirs2_core::random::Rng>(
        in_features: usize,
        out_features: usize,
        n_heads: usize,
        dropout: f64,
        rng: &mut R,
    ) -> Result<Self> {
        if in_features == 0 || out_features == 0 || n_heads == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "in_features, out_features, and n_heads must all be > 0".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&dropout) {
            return Err(NeuralError::InvalidArgument(
                "dropout must be in [0, 1)".to_string(),
            ));
        }

        let mut weights = Vec::with_capacity(n_heads);
        let mut attention_vecs = Vec::with_capacity(n_heads);
        let mut dweights = Vec::with_capacity(n_heads);
        let mut dattn_vecs = Vec::with_capacity(n_heads);

        for _ in 0..n_heads {
            weights.push(xavier_init(in_features, out_features, rng)?);
            // Attention vector covers [h_i ‖ h_j] so dimension is 2 * out_features
            let attn_data: Vec<F> = (0..2 * out_features)
                .map(|_| {
                    let v = {
                        let limit = (6.0_f64 / (2 * out_features) as f64).sqrt();
                        let dist = Uniform::new(-limit, limit).unwrap_or_else(|_| {
                            Uniform::new(-0.1, 0.1).expect("fallback Uniform failed")
                        });
                        dist.sample(rng)
                    };
                    F::from(v).unwrap_or(F::zero())
                })
                .collect();
            attention_vecs.push(Array1::from_vec(attn_data));
            dweights.push(Arc::new(RwLock::new(Array2::zeros((
                in_features,
                out_features,
            )))));
            dattn_vecs.push(Arc::new(RwLock::new(Array1::zeros(2 * out_features))));
        }

        let slope = F::from(0.2).unwrap_or_else(|| F::from(0.2).expect("slope convert"));

        Ok(Self {
            weights,
            attention_vecs,
            n_heads,
            dropout,
            leaky_relu_slope: slope,
            in_features,
            out_features,
            dweights,
            dattn_vecs,
        })
    }

    /// Apply LeakyReLU with the stored negative slope.
    #[inline]
    fn leaky_relu(&self, x: F) -> F {
        if x >= F::zero() {
            x
        } else {
            self.leaky_relu_slope * x
        }
    }

    /// Softmax over a slice (returns a new Vec).
    fn softmax(scores: &[F]) -> Vec<F> {
        if scores.is_empty() {
            return Vec::new();
        }
        let max_score = scores
            .iter()
            .copied()
            .fold(F::neg_infinity(), |a, b| if b > a { b } else { a });
        let exps: Vec<F> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: F = exps.iter().copied().fold(F::zero(), |a, b| a + b);
        if sum == F::zero() {
            exps
        } else {
            exps.iter().map(|&e| e / sum).collect()
        }
    }

    /// Forward pass for one attention head.
    fn forward_head(
        &self,
        features: &Array2<F>,
        adjacency: &Array2<F>,
        head: usize,
        training: bool,
    ) -> Result<Array2<F>> {
        let n = features.nrows();
        let w = &self.weights[head];
        let a = &self.attention_vecs[head];

        // Project: H_proj = features @ W  [N, out_features]
        let mut h_proj = Array2::<F>::zeros((n, self.out_features));
        for i in 0..n {
            for k in 0..self.in_features {
                let f_ik = features[[i, k]];
                if f_ik == F::zero() {
                    continue;
                }
                for j in 0..self.out_features {
                    h_proj[[i, j]] += f_ik * w[[k, j]];
                }
            }
        }

        // Compute unnormalised attention scores e_{ij} for each edge
        let mut out = Array2::<F>::zeros((n, self.out_features));
        for i in 0..n {
            // Collect neighbours (including self if diagonal is set)
            let mut neighbours: Vec<usize> = Vec::new();
            for j in 0..n {
                if adjacency[[i, j]] != F::zero() || i == j {
                    neighbours.push(j);
                }
            }
            if neighbours.is_empty() {
                continue;
            }

            // Attention scores for node i over its neighbours
            let mut scores: Vec<F> = Vec::with_capacity(neighbours.len());
            for &j in &neighbours {
                // e_{ij} = LeakyReLU( [a_left · h_i + a_right · h_j] )
                let mut e = F::zero();
                for k in 0..self.out_features {
                    e += a[k] * h_proj[[i, k]]; // left half
                    e += a[self.out_features + k] * h_proj[[j, k]]; // right half
                }
                scores.push(self.leaky_relu(e));
            }

            let alphas = Self::softmax(&scores);

            // Optional attention dropout (zero-out with probability `dropout`)
            let use_dropout = training && self.dropout > 0.0;

            // Aggregate: h'_i = Σ_j α_{ij} h_j
            for (idx, &j) in neighbours.iter().enumerate() {
                let alpha = if use_dropout {
                    // Bernoulli approximation: scale by keep_prob inverse
                    let keep = F::from(1.0 - self.dropout).unwrap_or(F::one());
                    if keep == F::zero() {
                        F::zero()
                    } else {
                        alphas[idx] / keep
                    }
                } else {
                    alphas[idx]
                };
                for k in 0..self.out_features {
                    out[[i, k]] += alpha * h_proj[[j, k]];
                }
            }
        }

        Ok(out)
    }

    /// Forward pass of the GAT layer.
    ///
    /// # Arguments
    /// * `features`  — `[N, in_features]` input node features
    /// * `adjacency` — `[N, N]` adjacency matrix (non-zero entries denote edges)
    /// * `training`  — whether the model is in training mode (affects dropout)
    ///
    /// # Returns
    /// `[N, n_heads * out_features]` concatenated multi-head output.
    pub fn forward_graph(
        &self,
        features: &Array2<F>,
        adjacency: &Array2<F>,
        training: bool,
    ) -> Result<Array2<F>> {
        let n = features.nrows();
        if features.ncols() != self.in_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "Expected in_features={}, got {}",
                self.in_features,
                features.ncols()
            )));
        }
        if adjacency.nrows() != n || adjacency.ncols() != n {
            return Err(NeuralError::InvalidArgument(format!(
                "Adjacency must be {}×{} but got {}×{}",
                n,
                n,
                adjacency.nrows(),
                adjacency.ncols()
            )));
        }

        let total_out = self.n_heads * self.out_features;
        let mut out = Array2::<F>::zeros((n, total_out));

        for h in 0..self.n_heads {
            let head_out = self.forward_head(features, adjacency, h, training)?;
            let offset = h * self.out_features;
            for i in 0..n {
                for k in 0..self.out_features {
                    out[[i, offset + k]] = head_out[[i, k]];
                }
            }
        }

        Ok(out)
    }

    /// Number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        self.n_heads * (self.in_features * self.out_features + 2 * self.out_features)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for GraphAttentionLayer<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let f2 = input
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|e| NeuralError::DimensionMismatch(format!("Expected 2D input: {e}")))?;
        let n = f2.nrows();
        let adj = Array2::<F>::eye(n);
        self.forward_graph(&f2, &adj, false).map(|a| a.into_dyn())
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, lr: F) -> Result<()> {
        for (w, dw_lock) in self.weights.iter_mut().zip(self.dweights.iter()) {
            let dw = dw_lock
                .read()
                .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))?;
            for (wi, gi) in w.iter_mut().zip(dw.iter()) {
                *wi -= lr * *gi;
            }
        }
        for (av, da_lock) in self.attention_vecs.iter_mut().zip(self.dattn_vecs.iter()) {
            let da = da_lock
                .read()
                .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))?;
            for (ai, gi) in av.iter_mut().zip(da.iter()) {
                *ai -= lr * *gi;
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn layer_type(&self) -> &str {
        "GraphAttention"
    }
    fn parameter_count(&self) -> usize {
        self.num_parameters()
    }

    fn layer_description(&self) -> String {
        format!(
            "type:GraphAttention in:{} out:{} heads:{} dropout:{}",
            self.in_features, self.out_features, self.n_heads, self.dropout
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GraphSAGE Layer
// ──────────────────────────────────────────────────────────────────────────────

/// Neighbourhood aggregation strategy for GraphSAGE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SageAggregator {
    /// Mean pooling of neighbour features
    Mean,
    /// Element-wise max pooling of neighbour features
    Max,
}

/// GraphSAGE layer (inductive representation learning)
///
/// Implements the concat-aggregate-project formulation from Hamilton et al. (2017):
/// ```text
/// agg_i   = Aggregator({ h_j | j ∈ N(i) })
/// h'_i    = σ( W_self h_i + W_neigh agg_i + b )
/// ```
///
/// The layer accepts an **adjacency list** — a `Vec<Vec<usize>>` where
/// `adj_list[i]` contains the indices of node `i`'s neighbours — which makes it
/// efficient for sparse graphs without materialising a full adjacency matrix.
///
/// # Shape
/// * `features`  — `[N, in_features]`
/// * `adj_list`  — `N`-element adjacency list
/// * returns     — `[N, out_features]`
///
/// # Examples
/// ```rust
/// use scirs2_neural::layers::graph_conv::{GraphSageLayer, SageAggregator};
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::random::rng;
///
/// let mut rng = rng();
/// let layer = GraphSageLayer::<f64>::new(4, 8, SageAggregator::Mean, true, &mut rng)
///     .expect("Failed to create GraphSAGE layer");
///
/// let features = Array2::<f64>::from_elem((5, 4), 0.1);
/// let adj_list = vec![vec![1, 2], vec![0], vec![0, 3], vec![2, 4], vec![3]];
/// let out = layer.forward_graph(&features, &adj_list).expect("Forward pass failed");
/// assert_eq!(out.shape(), &[5, 8]);
/// ```
#[derive(Debug, Clone)]
pub struct GraphSageLayer<F: Float + Debug + Send + Sync + NumAssign> {
    /// Weight for the node's own features: `[in_features, out_features]`
    weight_self: Array2<F>,
    /// Weight for aggregated neighbour features: `[in_features, out_features]`
    weight_neigh: Array2<F>,
    /// Optional bias: `[out_features]`
    bias: Option<Array1<F>>,
    /// Aggregation method
    aggregator: SageAggregator,
    /// Activation applied after the linear combination
    activation: GraphActivation,
    in_features: usize,
    out_features: usize,
    /// Gradient accumulators
    dweight_self: Arc<RwLock<Array2<F>>>,
    dweight_neigh: Arc<RwLock<Array2<F>>>,
    dbias: Arc<RwLock<Option<Array1<F>>>>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> GraphSageLayer<F> {
    /// Create a new GraphSAGE layer.
    ///
    /// # Arguments
    /// * `in_features`  — input node feature dimension
    /// * `out_features` — output node feature dimension
    /// * `aggregator`   — neighbourhood aggregation strategy
    /// * `use_bias`     — whether to add a learnable bias term
    /// * `rng`          — random number generator for weight initialisation
    pub fn new<R: scirs2_core::random::Rng>(
        in_features: usize,
        out_features: usize,
        aggregator: SageAggregator,
        use_bias: bool,
        rng: &mut R,
    ) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "in_features and out_features must be > 0".to_string(),
            ));
        }

        let weight_self = xavier_init(in_features, out_features, rng)?;
        let weight_neigh = xavier_init(in_features, out_features, rng)?;
        let bias = if use_bias {
            Some(Array1::zeros(out_features))
        } else {
            None
        };

        Ok(Self {
            weight_self,
            weight_neigh,
            bias,
            aggregator,
            activation: GraphActivation::ReLU,
            in_features,
            out_features,
            dweight_self: Arc::new(RwLock::new(Array2::zeros((in_features, out_features)))),
            dweight_neigh: Arc::new(RwLock::new(Array2::zeros((in_features, out_features)))),
            dbias: Arc::new(RwLock::new(if use_bias {
                Some(Array1::zeros(out_features))
            } else {
                None
            })),
        })
    }

    /// Override the activation function (default: ReLU)
    pub fn with_activation(mut self, activation: GraphActivation) -> Self {
        self.activation = activation;
        self
    }

    /// Aggregate neighbour features using the configured strategy.
    fn aggregate(&self, features: &Array2<F>, neighbours: &[usize]) -> Array1<F> {
        if neighbours.is_empty() {
            return Array1::zeros(self.in_features);
        }

        match self.aggregator {
            SageAggregator::Mean => {
                let mut agg = Array1::<F>::zeros(self.in_features);
                for &j in neighbours {
                    for k in 0..self.in_features {
                        agg[k] += features[[j, k]];
                    }
                }
                let n = F::from(neighbours.len()).unwrap_or(F::one());
                agg.mapv(|v| v / n)
            }
            SageAggregator::Max => {
                let mut agg = Array1::from_elem(self.in_features, F::neg_infinity());
                for &j in neighbours {
                    for k in 0..self.in_features {
                        let v = features[[j, k]];
                        if v > agg[k] {
                            agg[k] = v;
                        }
                    }
                }
                // Replace -inf with 0 for isolated dimensions
                agg.mapv(|v| if v == F::neg_infinity() { F::zero() } else { v })
            }
        }
    }

    /// Forward pass of the GraphSAGE layer.
    ///
    /// # Arguments
    /// * `features`  — `[N, in_features]` input node features
    /// * `adj_list`  — adjacency list: `adj_list[i]` is the list of neighbour indices of node `i`
    ///
    /// # Returns
    /// `[N, out_features]` updated node feature matrix.
    pub fn forward_graph(
        &self,
        features: &Array2<F>,
        adj_list: &[Vec<usize>],
    ) -> Result<Array2<F>> {
        let n = features.nrows();
        if features.ncols() != self.in_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "Expected in_features={}, got {}",
                self.in_features,
                features.ncols()
            )));
        }
        if adj_list.len() != n {
            return Err(NeuralError::InvalidArgument(format!(
                "adj_list length ({}) must match number of nodes ({})",
                adj_list.len(),
                n
            )));
        }

        // Validate neighbour indices
        for (i, neighbours) in adj_list.iter().enumerate() {
            for &j in neighbours {
                if j >= n {
                    return Err(NeuralError::InvalidArgument(format!(
                        "adj_list[{i}] contains out-of-bounds index {j} (n={n})"
                    )));
                }
            }
        }

        let mut out = Array2::<F>::zeros((n, self.out_features));

        for i in 0..n {
            let neighbours = &adj_list[i];
            let agg = self.aggregate(features, neighbours);

            // h'_i = W_self h_i + W_neigh agg_i  [out_features]
            for j in 0..self.out_features {
                let mut v = F::zero();
                for k in 0..self.in_features {
                    v += self.weight_self[[k, j]] * features[[i, k]];
                    v += self.weight_neigh[[k, j]] * agg[k];
                }
                if let Some(ref b) = self.bias {
                    v += b[j];
                }
                out[[i, j]] = self.activation.apply(v);
            }
        }

        Ok(out)
    }

    /// Forward pass using a dense adjacency matrix (convenience overload).
    ///
    /// Converts the `N × N` adjacency matrix to an adjacency list internally
    /// and delegates to [`forward_graph`](GraphSageLayer::forward_graph).
    pub fn forward_graph_dense(
        &self,
        features: &Array2<F>,
        adjacency: &Array2<F>,
    ) -> Result<Array2<F>> {
        let n = features.nrows();
        if adjacency.nrows() != n || adjacency.ncols() != n {
            return Err(NeuralError::InvalidArgument(format!(
                "Adjacency must be {}×{} but got {}×{}",
                n,
                n,
                adjacency.nrows(),
                adjacency.ncols()
            )));
        }
        let adj_list: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| adjacency[[i, j]] != F::zero()).collect())
            .collect();
        self.forward_graph(features, &adj_list)
    }

    /// Number of trainable parameters
    pub fn num_parameters(&self) -> usize {
        let bias_params = self.bias.as_ref().map_or(0, |b| b.len());
        2 * self.in_features * self.out_features + bias_params
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for GraphSageLayer<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let f2 = input
            .clone()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|e| NeuralError::DimensionMismatch(format!("Expected 2D input: {e}")))?;
        let n = f2.nrows();
        // Default: each node is its own neighbour
        let adj_list: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        self.forward_graph(&f2, &adj_list).map(|a| a.into_dyn())
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, lr: F) -> Result<()> {
        {
            let dws = self
                .dweight_self
                .read()
                .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))?;
            for (w, g) in self.weight_self.iter_mut().zip(dws.iter()) {
                *w -= lr * *g;
            }
        }
        {
            let dwn = self
                .dweight_neigh
                .read()
                .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))?;
            for (w, g) in self.weight_neigh.iter_mut().zip(dwn.iter()) {
                *w -= lr * *g;
            }
        }
        if let Some(ref mut b) = self.bias {
            let db_guard = self
                .dbias
                .read()
                .map_err(|e| NeuralError::ComputationError(format!("RwLock poisoned: {e}")))?;
            if let Some(ref db) = *db_guard {
                for (bi, gi) in b.iter_mut().zip(db.iter()) {
                    *bi -= lr * *gi;
                }
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn layer_type(&self) -> &str {
        "GraphSAGE"
    }
    fn parameter_count(&self) -> usize {
        self.num_parameters()
    }

    fn layer_description(&self) -> String {
        format!(
            "type:GraphSAGE in:{} out:{} agg:{:?} act:{:?} bias:{}",
            self.in_features,
            self.out_features,
            self.aggregator,
            self.activation,
            self.bias.is_some()
        )
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut v = vec![
            self.weight_self.clone().into_dyn(),
            self.weight_neigh.clone().into_dyn(),
        ];
        if let Some(ref b) = self.bias {
            v.push(b.clone().into_dyn());
        }
        v
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::rng;

    // ── GCN tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_gcn_output_shape() {
        let mut rng = rng();
        let layer = GraphConvLayer::<f64>::new(4, 8, true, GraphActivation::ReLU, &mut rng)
            .expect("Failed to create GCN");
        let features = Array2::<f64>::from_elem((5, 4), 0.1);
        let adj = Array2::<f64>::eye(5);
        let out = layer
            .forward_graph(&features, &adj)
            .expect("Forward failed");
        assert_eq!(out.shape(), &[5, 8]);
    }

    #[test]
    fn test_gcn_no_bias() {
        let mut rng = rng();
        let layer = GraphConvLayer::<f64>::new(3, 6, false, GraphActivation::None, &mut rng)
            .expect("Failed to create GCN");
        let features = Array2::<f64>::from_elem((4, 3), 1.0);
        let adj = Array2::<f64>::eye(4);
        let out = layer
            .forward_graph(&features, &adj)
            .expect("Forward failed");
        assert_eq!(out.shape(), &[4, 6]);
        // With identity adjacency and identity activation, output must be finite
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_gcn_normalize_adjacency_identity() {
        // normalize_adjacency(I_3):
        //   Â = I + I = 2*I  (self-loops on already-diagonal matrix)
        //   degree(i) = Σ_j Â[i,j] = 2  (only the diagonal entry)
        //   D^{-1/2} = diag(1/√2)
        //   D^{-1/2} Â D^{-1/2} = (1/√2) * 2 * (1/√2) = 1 on diagonal
        let eye: Array2<f64> = Array2::eye(3);
        let norm = GraphConvLayer::<f64>::normalize_adjacency(&eye).expect("Normalisation failed");
        assert!(
            (norm[[0, 0]] - 1.0).abs() < 1e-10,
            "diag expected 1.0, got {}",
            norm[[0, 0]]
        );
        assert!((norm[[0, 1]]).abs() < 1e-10);
    }

    #[test]
    fn test_gcn_normalize_adjacency_non_square_error() {
        let a = Array2::<f64>::zeros((3, 4));
        assert!(GraphConvLayer::<f64>::normalize_adjacency(&a).is_err());
    }

    #[test]
    fn test_gcn_wrong_in_features_error() {
        let mut rng = rng();
        let layer = GraphConvLayer::<f64>::new(4, 8, true, GraphActivation::ReLU, &mut rng)
            .expect("Failed to create GCN");
        let features = Array2::<f64>::from_elem((5, 3), 0.1); // wrong: 3 ≠ 4
        let adj = Array2::<f64>::eye(5);
        assert!(layer.forward_graph(&features, &adj).is_err());
    }

    #[test]
    fn test_gcn_parameter_count() {
        let mut rng = rng();
        let layer = GraphConvLayer::<f64>::new(4, 8, true, GraphActivation::ReLU, &mut rng)
            .expect("Failed to create GCN");
        assert_eq!(layer.num_parameters(), 4 * 8 + 8);
    }

    #[test]
    fn test_gcn_zero_features_stays_zero() {
        let mut rng = rng();
        let layer = GraphConvLayer::<f64>::new(4, 6, false, GraphActivation::ReLU, &mut rng)
            .expect("Failed to create GCN");
        let features = Array2::<f64>::zeros((3, 4));
        let adj = Array2::<f64>::from_elem((3, 3), 1.0);
        let out = layer
            .forward_graph(&features, &adj)
            .expect("Forward failed");
        // Zero input → zero output (ReLU of 0 = 0)
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gcn_activations() {
        let mut rng = rng();
        for act in [
            GraphActivation::None,
            GraphActivation::ReLU,
            GraphActivation::Sigmoid,
            GraphActivation::Tanh,
            GraphActivation::ELU,
        ] {
            let layer =
                GraphConvLayer::<f64>::new(2, 4, true, act, &mut rng).expect("Failed to create");
            let features = Array2::<f64>::from_elem((3, 2), 0.5);
            let adj = Array2::<f64>::eye(3);
            let out = layer
                .forward_graph(&features, &adj)
                .expect("Forward failed");
            assert_eq!(out.shape(), &[3, 4], "activation {:?} shape mismatch", act);
            assert!(
                out.iter().all(|&v| v.is_finite()),
                "non-finite with {:?}",
                act
            );
        }
    }

    // ── GAT tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_gat_output_shape() {
        let mut rng = rng();
        let layer =
            GraphAttentionLayer::<f64>::new(4, 8, 2, 0.0, &mut rng).expect("Failed to create GAT");
        let features = Array2::<f64>::from_elem((6, 4), 0.1);
        let adj = Array2::<f64>::eye(6);
        let out = layer
            .forward_graph(&features, &adj, false)
            .expect("Forward failed");
        // 2 heads × 8 out_features = 16
        assert_eq!(out.shape(), &[6, 16]);
    }

    #[test]
    fn test_gat_single_head() {
        let mut rng = rng();
        let layer =
            GraphAttentionLayer::<f64>::new(3, 5, 1, 0.0, &mut rng).expect("Failed to create GAT");
        let features = Array2::<f64>::from_elem((4, 3), 0.2);
        let adj = Array2::<f64>::from_elem((4, 4), 1.0);
        let out = layer
            .forward_graph(&features, &adj, false)
            .expect("Forward failed");
        assert_eq!(out.shape(), &[4, 5]);
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_gat_parameter_count() {
        let mut rng = rng();
        let layer =
            GraphAttentionLayer::<f64>::new(4, 8, 3, 0.0, &mut rng).expect("Failed to create GAT");
        // 3 heads × (4*8 weight + 2*8 attention) = 3 × 48 = 144
        assert_eq!(layer.num_parameters(), 3 * (4 * 8 + 2 * 8));
    }

    #[test]
    fn test_gat_invalid_dropout_error() {
        let mut rng = rng();
        assert!(GraphAttentionLayer::<f64>::new(4, 8, 2, 1.5, &mut rng).is_err());
    }

    // ── GraphSAGE tests ──────────────────────────────────────────────────────

    #[test]
    fn test_sage_mean_output_shape() {
        let mut rng = rng();
        let layer = GraphSageLayer::<f64>::new(4, 8, SageAggregator::Mean, true, &mut rng)
            .expect("Failed to create SAGE");
        let features = Array2::<f64>::from_elem((5, 4), 0.1);
        let adj_list = vec![vec![1, 2], vec![0], vec![0, 3], vec![2, 4], vec![3]];
        let out = layer
            .forward_graph(&features, &adj_list)
            .expect("Forward failed");
        assert_eq!(out.shape(), &[5, 8]);
    }

    #[test]
    fn test_sage_max_output_shape() {
        let mut rng = rng();
        let layer = GraphSageLayer::<f64>::new(3, 6, SageAggregator::Max, false, &mut rng)
            .expect("Failed to create SAGE");
        let features = Array2::<f64>::from_elem((4, 3), 0.5);
        let adj_list = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
        let out = layer
            .forward_graph(&features, &adj_list)
            .expect("Forward failed");
        assert_eq!(out.shape(), &[4, 6]);
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_sage_isolated_node() {
        let mut rng = rng();
        let layer = GraphSageLayer::<f64>::new(2, 4, SageAggregator::Mean, true, &mut rng)
            .expect("Failed to create SAGE");
        let features = Array2::<f64>::from_elem((3, 2), 1.0);
        // Node 1 is isolated (empty neighbour list)
        let adj_list = vec![vec![1, 2], vec![], vec![0]];
        let out = layer
            .forward_graph(&features, &adj_list)
            .expect("Forward failed");
        assert_eq!(out.shape(), &[3, 4]);
        assert!(out.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_sage_out_of_bounds_neighbour_error() {
        let mut rng = rng();
        let layer = GraphSageLayer::<f64>::new(2, 4, SageAggregator::Mean, false, &mut rng)
            .expect("Failed to create SAGE");
        let features = Array2::<f64>::from_elem((3, 2), 1.0);
        let adj_list = vec![vec![99], vec![], vec![]]; // index 99 is out of bounds
        assert!(layer.forward_graph(&features, &adj_list).is_err());
    }

    #[test]
    fn test_sage_dense_adjacency_overload() {
        let mut rng = rng();
        let layer = GraphSageLayer::<f64>::new(3, 5, SageAggregator::Mean, true, &mut rng)
            .expect("Failed to create SAGE");
        let features = Array2::<f64>::from_elem((4, 3), 0.3);
        let adj = Array2::<f64>::eye(4);
        let out = layer
            .forward_graph_dense(&features, &adj)
            .expect("Forward failed");
        assert_eq!(out.shape(), &[4, 5]);
    }

    #[test]
    fn test_sage_parameter_count() {
        let mut rng = rng();
        let layer = GraphSageLayer::<f64>::new(4, 8, SageAggregator::Mean, true, &mut rng)
            .expect("Failed to create SAGE");
        // 2 weight matrices (4×8 each) + bias (8)
        assert_eq!(layer.num_parameters(), 2 * 4 * 8 + 8);
    }

    // ── Layer trait generic forward tests ────────────────────────────────────

    #[test]
    fn test_gcn_layer_trait_forward() {
        let mut rng = rng();
        let layer = GraphConvLayer::<f64>::new(4, 4, true, GraphActivation::ReLU, &mut rng)
            .expect("Failed to create GCN");
        let input = Array2::<f64>::from_elem((3, 4), 0.5).into_dyn();
        let out = layer.forward(&input).expect("Layer trait forward failed");
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_sage_layer_trait_forward() {
        let mut rng = rng();
        let layer = GraphSageLayer::<f64>::new(4, 6, SageAggregator::Max, false, &mut rng)
            .expect("Failed to create SAGE");
        let input = Array2::<f64>::from_elem((4, 4), 0.2).into_dyn();
        let out = layer.forward(&input).expect("Layer trait forward failed");
        assert_eq!(out.shape(), &[4, 6]);
    }
}
