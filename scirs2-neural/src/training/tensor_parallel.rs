//! Tensor parallelism primitives: column-parallel, row-parallel linear layers,
//! and a vocabulary-partitioned parallel embedding.
//!
//! These components simulate the Megatron-LM tensor-parallel strategy
//! (Shoeybi et al., 2019) in a single-process environment by keeping
//! separate weight slices per "worker" in memory.
//!
//! ## Column Parallel Linear
//!
//! Splits the output dimension across workers.  Each worker computes
//! `y_i = x @ W_i + b_i` where `W_i = W[:, i*chunk:(i+1)*chunk]`.
//! The results are all-gathered (concatenated) to form the full output.
//!
//! ## Row Parallel Linear
//!
//! Splits the input dimension across workers.  Each worker handles
//! `x_i = x[:, i*chunk:(i+1)*chunk]` and computes `y_i = x_i @ W_i`.
//! An all-reduce (sum) combines the partial results, then the shared bias is added.
//!
//! ## Parallel Embedding
//!
//! Partitions the vocabulary across workers.  Each token index is routed to
//! the responsible worker; the resulting row is returned.
//!
//! ```rust
//! use scirs2_neural::training::tensor_parallel::{
//!     TensorParallelConfig, ColumnParallelLinear, RowParallelLinear, ParallelEmbedding,
//! };
//!
//! let cfg = TensorParallelConfig::default();
//! assert_eq!(cfg.n_workers, 2);
//!
//! let col = ColumnParallelLinear::new(8, 4, cfg.clone(), 0).expect("ok");
//! let input = scirs2_core::ndarray::Array2::<f64>::ones((3, 8));
//! let out = col.forward(&input).expect("ok");
//! assert_eq!(out.shape(), [3, 4]);
//! ```

use crate::error::{NeuralError, Result as NeuralResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

// ============================================================================
// Config
// ============================================================================

/// Configuration for tensor-parallel layers.
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Number of simulated workers.  Default: `2`.
    pub n_workers: usize,
    /// If `true`, all-gather the per-worker outputs after column-parallel linear.
    /// Default: `true`.
    pub gather_output: bool,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            n_workers: 2,
            gather_output: true,
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Xavier (Glorot) uniform initialisation scaled by `sqrt(2 / (fan_in + fan_out))`.
fn xavier_init(rng: &mut SmallRng, n_in: usize, n_out: usize) -> f64 {
    let scale = (6.0_f64 / (n_in + n_out) as f64).sqrt();
    rng.random::<f64>() * 2.0 * scale - scale
}

// ============================================================================
// ColumnParallelLinear
// ============================================================================

/// Splits the output dimension across `n_workers`.
///
/// With `n_workers = W` and output size `N`, each worker holds
/// weights of shape `[n_in, N/W]` and bias of shape `[N/W]`.
pub struct ColumnParallelLinear {
    /// Per-worker weight slices `[n_in, n_out/n_workers]`.
    local_weights: Vec<Array2<f64>>,
    /// Per-worker bias vectors `[n_out/n_workers]`.
    local_biases: Vec<Array1<f64>>,
    config: TensorParallelConfig,
    n_in: usize,
    total_n_out: usize,
}

impl ColumnParallelLinear {
    /// Create a column-parallel linear layer.
    ///
    /// # Errors
    /// - `n_out` is not divisible by `config.n_workers`.
    /// - `config.n_workers == 0`.
    pub fn new(
        n_in: usize,
        n_out: usize,
        config: TensorParallelConfig,
        seed: u64,
    ) -> NeuralResult<Self> {
        if config.n_workers == 0 {
            return Err(NeuralError::ConfigError(
                "TensorParallelConfig.n_workers must be > 0".into(),
            ));
        }
        if n_out % config.n_workers != 0 {
            return Err(NeuralError::ConfigError(format!(
                "n_out ({n_out}) must be divisible by n_workers ({})",
                config.n_workers
            )));
        }

        let chunk = n_out / config.n_workers;
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut local_weights = Vec::with_capacity(config.n_workers);
        let mut local_biases = Vec::with_capacity(config.n_workers);

        for _ in 0..config.n_workers {
            let w = Array2::from_shape_fn((n_in, chunk), |_| xavier_init(&mut rng, n_in, n_out));
            let b = Array1::zeros(chunk);
            local_weights.push(w);
            local_biases.push(b);
        }

        Ok(Self {
            local_weights,
            local_biases,
            config,
            n_in,
            total_n_out: n_out,
        })
    }

    /// Forward pass.
    ///
    /// Each worker computes `y_i = input @ W_i + b_i`.  If `gather_output`,
    /// the results are concatenated to `[batch, n_out]`; otherwise only the
    /// first worker's output is returned (for single-process simulation with
    /// `gather_output = false`).
    pub fn forward(&self, input: &Array2<f64>) -> NeuralResult<Array2<f64>> {
        let batch = input.shape()[0];
        let n_in = input.shape()[1];
        if n_in != self.n_in {
            return Err(NeuralError::DimensionMismatch(format!(
                "ColumnParallelLinear: expected n_in={}, got {n_in}",
                self.n_in
            )));
        }

        let mut parts: Vec<Array2<f64>> = Vec::with_capacity(self.config.n_workers);
        for (w, b) in self.local_weights.iter().zip(self.local_biases.iter()) {
            let y = input.dot(w) + b; // [batch, chunk]
            parts.push(y);
        }

        if self.config.gather_output {
            // Concatenate along feature axis.
            let chunk = self.total_n_out / self.config.n_workers;
            let mut gathered = Array2::<f64>::zeros((batch, self.total_n_out));
            for (wi, part) in parts.iter().enumerate() {
                let start = wi * chunk;
                let end = start + chunk;
                gathered.slice_mut(s![.., start..end]).assign(part);
            }
            Ok(gathered)
        } else {
            // Return first worker's slice.
            parts
                .into_iter()
                .next()
                .ok_or_else(|| NeuralError::ComputationError("no workers".into()))
        }
    }

    /// Total output features (after all-gather).
    pub fn n_out(&self) -> usize {
        self.total_n_out
    }

    /// Number of simulated workers.
    pub fn n_workers(&self) -> usize {
        self.config.n_workers
    }
}

// ============================================================================
// RowParallelLinear
// ============================================================================

/// Splits the input dimension across `n_workers`.
///
/// Each worker holds weights `[n_in/n_workers, n_out]`.  The partial results
/// are summed (all-reduce) and the shared bias is added once.
pub struct RowParallelLinear {
    /// Per-worker weight slices `[n_in/n_workers, n_out]`.
    local_weights: Vec<Array2<f64>>,
    /// Shared bias `[n_out]` (added after all-reduce).
    bias: Array1<f64>,
    config: TensorParallelConfig,
    total_n_in: usize,
    n_out: usize,
}

impl RowParallelLinear {
    /// Create a row-parallel linear layer.
    ///
    /// # Errors
    /// - `n_in` is not divisible by `config.n_workers`.
    /// - `config.n_workers == 0`.
    pub fn new(
        n_in: usize,
        n_out: usize,
        config: TensorParallelConfig,
        seed: u64,
    ) -> NeuralResult<Self> {
        if config.n_workers == 0 {
            return Err(NeuralError::ConfigError(
                "TensorParallelConfig.n_workers must be > 0".into(),
            ));
        }
        if n_in % config.n_workers != 0 {
            return Err(NeuralError::ConfigError(format!(
                "n_in ({n_in}) must be divisible by n_workers ({})",
                config.n_workers
            )));
        }

        let chunk = n_in / config.n_workers;
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut local_weights = Vec::with_capacity(config.n_workers);
        for _ in 0..config.n_workers {
            let w =
                Array2::from_shape_fn((chunk, n_out), |_| xavier_init(&mut rng, n_in, n_out));
            local_weights.push(w);
        }
        let bias = Array1::zeros(n_out);

        Ok(Self {
            local_weights,
            bias,
            config,
            total_n_in: n_in,
            n_out,
        })
    }

    /// Forward pass.
    ///
    /// Each worker computes `y_i = input_i @ W_i` where
    /// `input_i = input[:, i*chunk:(i+1)*chunk]`.
    /// The partial products are summed and the bias is added: `y = Σ y_i + bias`.
    pub fn forward(&self, input: &Array2<f64>) -> NeuralResult<Array2<f64>> {
        let batch = input.shape()[0];
        let n_in = input.shape()[1];
        if n_in != self.total_n_in {
            return Err(NeuralError::DimensionMismatch(format!(
                "RowParallelLinear: expected n_in={}, got {n_in}",
                self.total_n_in
            )));
        }

        let chunk = self.total_n_in / self.config.n_workers;
        let mut acc = Array2::<f64>::zeros((batch, self.n_out));

        for (wi, w) in self.local_weights.iter().enumerate() {
            let start = wi * chunk;
            let end = start + chunk;
            let input_slice = input.slice(s![.., start..end]);
            let partial = input_slice.dot(w); // [batch, n_out]
            acc += &partial;
        }

        // Add shared bias.
        acc += &self.bias;

        Ok(acc)
    }

    /// Total input features (across all workers).
    pub fn n_in(&self) -> usize {
        self.total_n_in
    }
}

// ============================================================================
// ParallelEmbedding
// ============================================================================

/// Vocabulary-partitioned embedding table.
///
/// The vocabulary is split evenly across `n_workers`.  Each token index is
/// routed to worker `index / (vocab_size / n_workers)` and the corresponding
/// row is returned.
pub struct ParallelEmbedding {
    /// Per-worker embedding sub-tables `[vocab_size/n_workers, embed_dim]`.
    local_tables: Vec<Array2<f64>>,
    vocab_size: usize,
    embed_dim: usize,
    n_workers: usize,
}

impl ParallelEmbedding {
    /// Create a parallel embedding table.
    ///
    /// # Errors
    /// - `vocab_size` is not divisible by `n_workers`.
    /// - `n_workers == 0`.
    pub fn new(
        vocab_size: usize,
        embed_dim: usize,
        n_workers: usize,
        seed: u64,
    ) -> NeuralResult<Self> {
        if n_workers == 0 {
            return Err(NeuralError::ConfigError(
                "ParallelEmbedding: n_workers must be > 0".into(),
            ));
        }
        if vocab_size % n_workers != 0 {
            return Err(NeuralError::ConfigError(format!(
                "vocab_size ({vocab_size}) must be divisible by n_workers ({n_workers})"
            )));
        }

        let local_vocab = vocab_size / n_workers;
        let mut rng = SmallRng::seed_from_u64(seed);

        // Small normal initialisation for embeddings.
        let mut local_tables = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let table = Array2::from_shape_fn((local_vocab, embed_dim), |_| {
                (rng.random::<f64>() * 2.0 - 1.0) * 0.02
            });
            local_tables.push(table);
        }

        Ok(Self {
            local_tables,
            vocab_size,
            embed_dim,
            n_workers,
        })
    }

    /// Look up embeddings for a sequence of token indices.
    ///
    /// Returns an array of shape `[len(indices), embed_dim]`.
    ///
    /// # Errors
    /// Returns `NeuralError::InvalidArgument` if any index >= `vocab_size`.
    pub fn forward(&self, indices: &[usize]) -> NeuralResult<Array2<f64>> {
        let local_vocab = self.vocab_size / self.n_workers;
        let mut out = Array2::<f64>::zeros((indices.len(), self.embed_dim));

        for (row, &idx) in indices.iter().enumerate() {
            if idx >= self.vocab_size {
                return Err(NeuralError::InvalidArgument(format!(
                    "token index {idx} out of range (vocab_size={})",
                    self.vocab_size
                )));
            }
            let worker_id = idx / local_vocab;
            let local_idx = idx % local_vocab;
            let embedding = self.local_tables[worker_id].slice(s![local_idx, ..]);
            out.slice_mut(s![row, ..]).assign(&embedding);
        }

        Ok(out)
    }

    /// Total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // --- TensorParallelConfig ---

    #[test]
    fn test_default_config_n_workers_2() {
        let cfg = TensorParallelConfig::default();
        assert_eq!(cfg.n_workers, 2, "default n_workers must be 2");
        assert!(cfg.gather_output, "default gather_output must be true");
    }

    // --- ColumnParallelLinear ---

    #[test]
    fn test_column_parallel_output_shape() {
        let cfg = TensorParallelConfig { n_workers: 2, gather_output: true };
        let layer = ColumnParallelLinear::new(8, 4, cfg, 0).expect("ok");
        let input = Array2::<f64>::ones((5, 8));
        let out = layer.forward(&input).expect("forward ok");
        assert_eq!(out.shape(), [5, 4], "output shape should be [batch, n_out]");
    }

    #[test]
    fn test_column_parallel_n_out() {
        let cfg = TensorParallelConfig { n_workers: 4, gather_output: true };
        let layer = ColumnParallelLinear::new(6, 8, cfg, 1).expect("ok");
        assert_eq!(layer.n_out(), 8);
        assert_eq!(layer.n_workers(), 4);
    }

    #[test]
    fn test_column_parallel_n_workers_1_equivalent_to_linear() {
        // With 1 worker, output should be same as a regular linear (W*X + b).
        let n_in = 4;
        let n_out = 6;
        let cfg = TensorParallelConfig { n_workers: 1, gather_output: true };
        let layer = ColumnParallelLinear::new(n_in, n_out, cfg, 42).expect("ok");
        let input = Array2::from_shape_fn((3, n_in), |(i, j)| (i * n_in + j) as f64 * 0.1);
        let out = layer.forward(&input).expect("forward ok");
        // Manual linear: y = input @ W + b.
        let expected = input.dot(&layer.local_weights[0]) + &layer.local_biases[0];
        let diff: f64 = (&out - &expected).mapv(|v| v.abs()).sum();
        assert!(diff < 1e-12, "n_workers=1 must match single linear; diff={diff}");
    }

    #[test]
    fn test_column_parallel_indivisible_n_out_error() {
        let cfg = TensorParallelConfig { n_workers: 3, gather_output: true };
        assert!(
            ColumnParallelLinear::new(4, 7, cfg, 0).is_err(),
            "n_out=7 is not divisible by 3"
        );
    }

    // --- RowParallelLinear ---

    #[test]
    fn test_row_parallel_output_shape() {
        let cfg = TensorParallelConfig { n_workers: 2, gather_output: true };
        let layer = RowParallelLinear::new(8, 4, cfg, 0).expect("ok");
        let input = Array2::<f64>::ones((5, 8));
        let out = layer.forward(&input).expect("forward ok");
        assert_eq!(out.shape(), [5, 4], "output shape should be [batch, n_out]");
    }

    #[test]
    fn test_row_parallel_n_in() {
        let cfg = TensorParallelConfig { n_workers: 2, gather_output: true };
        let layer = RowParallelLinear::new(6, 3, cfg, 0).expect("ok");
        assert_eq!(layer.n_in(), 6);
    }

    #[test]
    fn test_row_parallel_all_reduce_equals_full_matmul() {
        // Row-parallel sum across workers must equal a full matrix multiply.
        let n_in = 8;
        let n_out = 4;
        let cfg = TensorParallelConfig { n_workers: 2, gather_output: true };
        let layer = RowParallelLinear::new(n_in, n_out, cfg, 7).expect("ok");
        let input = Array2::from_shape_fn((3, n_in), |(i, j)| (i * n_in + j) as f64 * 0.1);
        let out_parallel = layer.forward(&input).expect("row parallel ok");

        // Reconstruct full weight by concatenating [W_0; W_1].
        use scirs2_core::ndarray::concatenate;
        use scirs2_core::ndarray::Axis;
        let full_w: Array2<f64> = concatenate(
            Axis(0),
            &[
                layer.local_weights[0].view(),
                layer.local_weights[1].view(),
            ],
        )
        .expect("concat ok");
        let out_full = input.dot(&full_w) + &layer.bias;

        let diff: f64 = (&out_parallel - &out_full)
            .mapv(|v| v.abs())
            .sum();
        assert!(diff < 1e-12, "row-parallel must equal full matmul; diff={diff}");
    }

    #[test]
    fn test_col_row_composition_shape() {
        let n_in = 8;
        let hidden = 16;
        let n_out = 4;
        let cfg1 = TensorParallelConfig { n_workers: 2, gather_output: true };
        let cfg2 = TensorParallelConfig { n_workers: 2, gather_output: true };
        let col = ColumnParallelLinear::new(n_in, hidden, cfg1, 0).expect("col ok");
        let row = RowParallelLinear::new(hidden, n_out, cfg2, 1).expect("row ok");
        let input = Array2::<f64>::ones((5, n_in));
        let mid = col.forward(&input).expect("col forward");
        let out = row.forward(&mid).expect("row forward");
        assert_eq!(out.shape(), [5, n_out]);
    }

    // --- ParallelEmbedding ---

    #[test]
    fn test_parallel_embedding_output_shape() {
        let emb = ParallelEmbedding::new(8, 16, 2, 0).expect("ok");
        let indices = vec![0_usize, 1, 3, 7];
        let out = emb.forward(&indices).expect("forward ok");
        assert_eq!(out.shape(), [4, 16], "shape should be [n_indices, embed_dim]");
    }

    #[test]
    fn test_parallel_embedding_vocab_and_dim() {
        let emb = ParallelEmbedding::new(100, 32, 4, 0).expect("ok");
        assert_eq!(emb.vocab_size(), 100);
        assert_eq!(emb.embed_dim(), 32);
    }

    #[test]
    fn test_parallel_embedding_same_index_same_vector() {
        let emb = ParallelEmbedding::new(8, 4, 2, 99).expect("ok");
        let out1 = emb.forward(&[3]).expect("ok");
        let out2 = emb.forward(&[3]).expect("ok");
        let diff: f64 = (&out1 - &out2).mapv(|v| v.abs()).sum();
        assert!(diff < 1e-15, "same index must always return same embedding");
    }

    #[test]
    fn test_parallel_embedding_out_of_range_error() {
        let emb = ParallelEmbedding::new(8, 4, 2, 0).expect("ok");
        assert!(
            emb.forward(&[8]).is_err(),
            "index 8 is out of range for vocab_size=8"
        );
    }

    #[test]
    fn test_parallel_embedding_indivisible_vocab_error() {
        assert!(
            ParallelEmbedding::new(7, 4, 2, 0).is_err(),
            "vocab_size=7 not divisible by 2"
        );
    }
}
