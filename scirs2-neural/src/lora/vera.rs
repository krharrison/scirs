//! VeRA: Vector-based Random Matrix Adaptation.
//!
//! VeRA shares a single pair of frozen random matrices (A, B) across all adapted
//! layers.  Each layer only learns two small scaling vectors:
//!
//! ```text
//! ΔW = diag(d) · B · diag(b) · A
//! ```
//!
//! where
//! - `A ∈ ℝ^[rank × in]`  — frozen random (regenerated from seed)
//! - `B ∈ ℝ^[out × rank]` — frozen random (regenerated from seed)
//! - `b ∈ ℝ^rank`          — **trainable** per-rank scaling
//! - `d ∈ ℝ^out`           — **trainable** per-output scaling
//!
//! The random matrices are never stored in memory; they are regenerated on each
//! call from a deterministic LCG seeded with `projection_seed`.  This allows
//! multiple layers to share the same logical A/B without any communication.
//!
//! This module provides two complementary APIs:
//!
//! * **`VeraLayer` / `VeraConfig`** — regenerate-on-call variant: A and B are
//!   recomputed from the seed each forward pass, avoiding storage entirely.
//!
//! * **`VeRALayer` / `VeRAConfig` / `SharedRandomMatrices`** — explicit-shared
//!   variant: A and B are materialised once and stored in a `SharedRandomMatrices`
//!   struct that can be cheaply cloned or shared across layers.
//!
//! # References
//!
//! - Kopiczko et al., "VeRA: Vector-Based Random Matrix Adaptation", ICLR 2024

use scirs2_core::ndarray::{Array1, Array2};

use crate::{NeuralError, Result};

// ══════════════════════════════════════════════════════════════════════════════
// VeraConfig / VeraLayer  (regenerate-on-call variant)
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for a VeRA layer (regenerate-on-call variant).
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::vera::VeraConfig;
///
/// let cfg = VeraConfig::default();
/// assert_eq!(cfg.rank, 64);
/// ```
#[derive(Debug, Clone)]
pub struct VeraConfig {
    /// Shared random-projection rank.  Larger rank = better expressiveness at
    /// the cost of memory for intermediate computations.
    pub rank: usize,
    /// Scaling factor α; effective scale = α/rank.
    pub alpha: f64,
    /// Seed used to regenerate the frozen random matrices A and B.
    /// All layers using the same seed share the same projections.
    pub projection_seed: u64,
}

impl Default for VeraConfig {
    fn default() -> Self {
        Self {
            rank: 64,
            alpha: 1.0,
            projection_seed: 42,
        }
    }
}

// ──────────────────────────── VeraLayer ──────────────────────────────────────

/// VeRA layer with frozen random projections and learned scaling vectors
/// (regenerate-on-call variant).
///
/// Trainable tensors: `b_scale` [rank], `d_scale` [out_features].
/// Frozen tensors: `weight` [out × in] (base), A and B (regenerated from seed).
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::vera::{VeraLayer, VeraConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let weight = Array2::<f64>::eye(8);
/// let cfg = VeraConfig { rank: 8, ..Default::default() };
/// let layer = VeraLayer::new(weight, &cfg).expect("create VeraLayer");
/// let input = Array2::<f64>::ones((2, 8));
/// let out = layer.forward(&input).expect("forward");
/// assert_eq!(out.shape(), &[2, 8]);
/// ```
pub struct VeraLayer {
    /// Frozen base weight [out × in].
    weight: Array2<f64>,
    /// Per-rank trainable scaling vector `b` [rank].
    pub b_scale: Array1<f64>,
    /// Per-output trainable scaling vector `d` [out_features].
    pub d_scale: Array1<f64>,
    out_features: usize,
    in_features: usize,
    /// Precomputed α/rank.
    scaling: f64,
    config: VeraConfig,
}

impl VeraLayer {
    /// Create a new VeRA layer.
    ///
    /// - `b_scale` initialised to `1 / sqrt(rank)`.
    /// - `d_scale` initialised to ones.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::InvalidArgument`] if `rank == 0`.
    pub fn new(weight: Array2<f64>, config: &VeraConfig) -> Result<Self> {
        if config.rank == 0 {
            return Err(NeuralError::InvalidArgument(
                "VeRA rank must be > 0".to_string(),
            ));
        }

        let (out_f, in_f) = (weight.nrows(), weight.ncols());
        let scaling = config.alpha / config.rank as f64;
        let b_init = 1.0 / (config.rank as f64).sqrt();

        Ok(Self {
            weight,
            b_scale: Array1::from_elem(config.rank, b_init),
            d_scale: Array1::ones(out_f),
            out_features: out_f,
            in_features: in_f,
            scaling,
            config: config.clone(),
        })
    }

    /// Generate the frozen random A matrix [rank × in] from the layer seed.
    ///
    /// Uses a 64-bit LCG (Knuth multiplicative hash) to avoid the `rand` crate.
    /// The distribution is uniform in `[−1, 1] / sqrt(rank)`.
    pub fn generate_a(&self) -> Array2<f64> {
        let mut state = self.config.projection_seed;
        let r = self.config.rank;
        let scale = 1.0 / (r as f64).sqrt();
        Array2::from_shape_fn((r, self.in_features), |_| {
            state = lcg_next(state);
            lcg_to_uniform(state) * scale
        })
    }

    /// Generate the frozen random B matrix [out × rank] from a different seed.
    ///
    /// The seed offset `0xDEAD_BEEF_DEAD_BEEF` ensures A and B are statistically
    /// independent even though both derive from `projection_seed`.
    pub fn generate_b(&self) -> Array2<f64> {
        let mut state = self
            .config
            .projection_seed
            .wrapping_add(0xDEAD_BEEF_DEAD_BEEF);
        let r = self.config.rank;
        let scale = 1.0 / (r as f64).sqrt();
        Array2::from_shape_fn((self.out_features, r), |_| {
            state = lcg_next(state);
            lcg_to_uniform(state) * scale
        })
    }

    /// Compute the weight delta: `scaling · diag(d) · B · diag(b) · A`.
    ///
    /// Steps:
    /// 1. `BA = diag(b) · A`  (scale rows of A by b_scale)
    /// 2. `delta = B · BA`
    /// 3. `delta *= scaling`
    /// 4. `delta = diag(d) · delta`  (scale rows of delta by d_scale)
    pub fn compute_delta(&self) -> Array2<f64> {
        let a = self.generate_a(); // [rank × in]
        let b_mat = self.generate_b(); // [out × rank]

        // diag(b) · A: scale each row i of A by b_scale[i]
        let mut scaled_a = a;
        for (i, mut row) in scaled_a.rows_mut().into_iter().enumerate() {
            let s = self.b_scale[i];
            row.mapv_inplace(|v| v * s);
        }

        // B · (diag(b) · A): [out × in]
        let mut delta = b_mat.dot(&scaled_a) * self.scaling;

        // diag(d) · delta: scale each row i by d_scale[i]
        for (i, mut row) in delta.rows_mut().into_iter().enumerate() {
            let s = self.d_scale[i];
            row.mapv_inplace(|v| v * s);
        }

        delta
    }

    /// Compute `W_eff = W_0 + compute_delta()`.
    pub fn effective_weight(&self) -> Array2<f64> {
        &self.weight + &self.compute_delta()
    }

    /// Forward pass: `y = x · W_eff^T`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::DimensionMismatch`] on input-shape mismatch.
    pub fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        if input.ncols() != self.in_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "VeRA expects {} input features, got {}",
                self.in_features,
                input.ncols()
            )));
        }
        Ok(input.dot(&self.effective_weight().t()))
    }

    /// Number of trainable parameters: `rank + out_features`.
    ///
    /// VeRA is much more parameter-efficient than LoRA of the same rank because
    /// it shares A and B across layers and only stores two small vectors.
    pub fn n_trainable_params(&self) -> usize {
        self.b_scale.len() + self.d_scale.len()
    }

    /// Dimension of the layer `(out_features, in_features)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }

    /// Reference to the frozen base weight.
    pub fn weight(&self) -> &Array2<f64> {
        &self.weight
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// VeRAConfig / SharedRandomMatrices / VeRALayer  (explicit-shared-matrix API)
// ══════════════════════════════════════════════════════════════════════════════

/// Configuration for a VeRA layer (explicit-shared-matrix variant).
///
/// Uses `rank`, `alpha`, and `seed` fields to match the canonical Kopiczko et al.
/// description.  The `seed` drives the deterministic LCG used to fill `A` and `B`.
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::vera::VeRAConfig;
///
/// let cfg = VeRAConfig::default();
/// assert_eq!(cfg.rank, 8);
/// assert_eq!(cfg.alpha, 16.0);
/// assert_eq!(cfg.seed, 42);
/// ```
#[derive(Debug, Clone)]
pub struct VeRAConfig {
    /// Shared random-projection rank.
    pub rank: usize,
    /// Scaling hyper-parameter α; effective scale = α / rank.
    pub alpha: f64,
    /// Seed used to generate the frozen random matrices A and B.
    /// All adapted layers that share this seed use identical projections.
    pub seed: u64,
}

impl Default for VeRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            seed: 42,
        }
    }
}

// ──────────────────────────── SharedRandomMatrices ───────────────────────────

/// Frozen shared random matrices used by VeRA layers.
///
/// Holds `a: Array2<f64>` of shape `[rank × in_features]` and
/// `b: Array2<f64>` of shape `[out_features × rank]`.  Both are generated
/// deterministically from `seed` using an LCG + Box–Muller PRNG, making them
/// reproducible without per-layer storage.
///
/// `b` is initialised to **zeros** so that ΔW = 0 at the start of training
/// (matching the LoRA zero-init convention on the second low-rank factor).
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::vera::SharedRandomMatrices;
///
/// let shared = SharedRandomMatrices::new(32, 64, 8, 42);
/// assert_eq!(shared.a.shape(), &[8, 32]);
/// assert_eq!(shared.b.shape(), &[64, 8]);
/// ```
#[derive(Debug, Clone)]
pub struct SharedRandomMatrices {
    /// Frozen A matrix  [rank × in_features], drawn from N(0, 1/√rank).
    pub a: Array2<f64>,
    /// Frozen B matrix  [out_features × rank], initialised to zeros.
    pub b: Array2<f64>,
}

impl SharedRandomMatrices {
    /// Build the shared frozen matrices for `(in_features, out_features, rank, seed)`.
    ///
    /// `a` is filled with samples from N(0, 1/√rank) via a Box–Muller LCG.
    /// `b` is initialised to zeros so the adapter is a no-op at init.
    pub fn new(in_features: usize, out_features: usize, rank: usize, seed: u64) -> Self {
        let a = Self::fill_normal(rank, in_features, seed, rank);
        let b = Array2::zeros((out_features, rank));
        Self { a, b }
    }

    /// Fill an `[rows × cols]` matrix with N(0, 1/√norm_rows) samples drawn from
    /// a deterministic LCG using the Box–Muller transform for Gaussian samples.
    fn fill_normal(rows: usize, cols: usize, seed: u64, norm_rows: usize) -> Array2<f64> {
        let sigma = 1.0 / (norm_rows as f64).sqrt();
        let mut state = if seed == 0 {
            0xCAFE_BABE_DEAD_BEEF
        } else {
            seed
        };
        let total = rows * cols;

        // Box–Muller requires pairs; accumulate into a flat Vec then reshape.
        let mut values = Vec::with_capacity(total);
        let mut i = 0usize;
        while i < total {
            state = lcg_next(state);
            let u1 = lcg_to_unit(state);
            state = lcg_next(state);
            let u2 = lcg_to_unit(state);

            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            values.push(r * theta.cos() * sigma);
            i += 1;
            if i < total {
                values.push(r * theta.sin() * sigma);
                i += 1;
            }
        }

        Array2::from_shape_vec((rows, cols), values).unwrap_or_else(|_| Array2::zeros((rows, cols)))
    }
}

// ──────────────────────────── VeRALayer ──────────────────────────────────────

/// VeRA layer with explicit `SharedRandomMatrices` and per-layer scaling vectors.
///
/// The weight update is:
///
/// ```text
/// ΔW = diag(b_vec) · shared.b · diag(d) · shared.a · (α / rank)
/// ```
///
/// where `d ∈ ℝ^rank` and `b_vec ∈ ℝ^out_features` are the only trainable
/// parameters.  The shared frozen matrices `shared.a` and `shared.b` are stored
/// by value and can be cheaply cloned to share across multiple layers (or wrapped
/// in an `Arc` at the call site for zero-copy sharing).
///
/// # Parameter efficiency
///
/// `trainable_params = rank + out_features`  vs `rank·(in+out)` for LoRA.
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::vera::{VeRAConfig, VeRALayer, SharedRandomMatrices};
/// use scirs2_core::ndarray::Array2;
///
/// let weight = Array2::<f64>::eye(8);
/// let shared = SharedRandomMatrices::new(8, 8, 4, 42);
/// let cfg = VeRAConfig { rank: 4, alpha: 8.0, seed: 42 };
/// let layer = VeRALayer::new(weight, shared, &cfg).expect("create layer");
///
/// let input = Array2::<f64>::ones((3, 8));
/// let out = layer.forward(&input).expect("forward");
/// assert_eq!(out.shape(), &[3, 8]);
/// ```
pub struct VeRALayer {
    /// Frozen base weight [out_features × in_features].
    pub weight: Array2<f64>,
    /// Per-rank trainable scaling vector `d` [rank].
    pub d: Array1<f64>,
    /// Per-output trainable scaling vector `b_vec` [out_features].
    pub b_vec: Array1<f64>,
    /// Frozen shared random projections.
    pub shared: SharedRandomMatrices,
    /// Layer configuration.
    config: VeRAConfig,
    /// Whether the delta has been merged permanently into `weight`.
    merged: bool,
    out_features: usize,
    in_features: usize,
}

impl VeRALayer {
    /// Create a new VeRA layer.
    ///
    /// Validates that:
    /// - `shared.a` has shape `[rank × in_features]`
    /// - `shared.b` has shape `[out_features × rank]`
    ///
    /// `d` is initialised to `ones * 0.01`, `b_vec` to `ones * 0.01`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::DimensionMismatch`] if the shared matrices are
    /// incompatible with the weight dimensions or rank.
    pub fn new(
        weight: Array2<f64>,
        shared: SharedRandomMatrices,
        config: &VeRAConfig,
    ) -> Result<Self> {
        let out_f = weight.nrows();
        let in_f = weight.ncols();
        let rank = config.rank;

        if shared.a.nrows() != rank {
            return Err(NeuralError::DimensionMismatch(format!(
                "shared.a must have {} rows (rank), got {}",
                rank,
                shared.a.nrows()
            )));
        }
        if shared.a.ncols() != in_f {
            return Err(NeuralError::DimensionMismatch(format!(
                "shared.a must have {} cols (in_features={in_f}), got {}",
                in_f,
                shared.a.ncols()
            )));
        }
        if shared.b.nrows() != out_f {
            return Err(NeuralError::DimensionMismatch(format!(
                "shared.b must have {} rows (out_features={out_f}), got {}",
                out_f,
                shared.b.nrows()
            )));
        }
        if shared.b.ncols() != rank {
            return Err(NeuralError::DimensionMismatch(format!(
                "shared.b must have {} cols (rank), got {}",
                rank,
                shared.b.ncols()
            )));
        }

        Ok(Self {
            weight,
            d: Array1::from_elem(rank, 0.01),
            b_vec: Array1::from_elem(out_f, 0.01),
            shared,
            config: config.clone(),
            merged: false,
            out_features: out_f,
            in_features: in_f,
        })
    }

    /// Precomputed scaling factor: α / rank.
    #[inline]
    pub fn scaling(&self) -> f64 {
        self.config.alpha / self.config.rank as f64
    }

    /// Compute weight delta `diag(b_vec) · shared.b · diag(d) · shared.a · scaling`.
    ///
    /// Step-by-step:
    /// 1. `da  = diag(d) · A`     — scale rows of A by `d`        → [rank × in]
    /// 2. `bda = B · da`           — matrix multiply                → [out × in]
    /// 3. `bda *= scaling`
    /// 4. `diag(b_vec) · bda`     — scale rows by `b_vec`         → [out × in]
    pub fn delta_weight(&self) -> Array2<f64> {
        let a = &self.shared.a; // [rank × in]
        let b_mat = &self.shared.b; // [out × rank]
        let scaling = self.scaling();

        // 1. diag(d) · A: scale each row i of A by d[i]
        let mut scaled_a = a.clone();
        for (i, mut row) in scaled_a.rows_mut().into_iter().enumerate() {
            let s = self.d[i];
            row.mapv_inplace(|v| v * s);
        }

        // 2 + 3. B · (diag(d) · A) * scaling → [out × in]
        let mut delta = b_mat.dot(&scaled_a) * scaling;

        // 4. diag(b_vec) · delta: scale each row i by b_vec[i]
        for (i, mut row) in delta.rows_mut().into_iter().enumerate() {
            let s = self.b_vec[i];
            row.mapv_inplace(|v| v * s);
        }

        delta
    }

    /// Effective weight: `W_0 + delta_weight()` (or just `W_0` when merged).
    pub fn effective_weight(&self) -> Array2<f64> {
        if self.merged {
            self.weight.clone()
        } else {
            &self.weight + &self.delta_weight()
        }
    }

    /// Merge `delta_weight()` permanently into `weight` and set `merged = true`.
    ///
    /// After merging, `forward` is equivalent to a plain linear layer with no
    /// overhead from computing the delta.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::InvalidArgument`] if already merged.
    pub fn merge(&mut self) -> Result<()> {
        if self.merged {
            return Err(NeuralError::InvalidArgument(
                "VeRALayer is already merged".to_string(),
            ));
        }
        let delta = self.delta_weight();
        self.weight = &self.weight + &delta;
        self.merged = true;
        Ok(())
    }

    /// Unmerge: subtract `delta_weight()` from `weight` and set `merged = false`.
    ///
    /// Exact only when `d` and `b_vec` have not changed since `merge()`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::InvalidArgument`] if not currently merged.
    pub fn unmerge(&mut self) -> Result<()> {
        if !self.merged {
            return Err(NeuralError::InvalidArgument(
                "VeRALayer is not merged".to_string(),
            ));
        }
        // Mark as unmerged first so delta_weight() uses the base formulas.
        self.merged = false;
        let delta = self.delta_weight();
        self.weight = &self.weight - &delta;
        Ok(())
    }

    /// Forward pass: `y = effective_weight() · x^T`, transposed to `[batch × out]`.
    ///
    /// Accepts `input` of shape `[batch × in_features]`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::DimensionMismatch`] if `input.ncols() != in_features`.
    pub fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        if input.ncols() != self.in_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "VeRALayer expects {} input features, got {}",
                self.in_features,
                input.ncols()
            )));
        }
        // W_eff [out × in] · input^T [in × batch] = output [out × batch]
        // transpose → [batch × out]
        Ok(self.effective_weight().dot(&input.t()).t().to_owned())
    }

    /// Number of trainable parameters: `rank + out_features`.
    pub fn trainable_params(&self) -> usize {
        self.d.len() + self.b_vec.len()
    }

    /// Total parameters in the base weight matrix: `out_features × in_features`.
    pub fn total_params(&self) -> usize {
        self.weight.len()
    }

    /// Whether the adapter has been merged into the base weight.
    pub fn is_merged(&self) -> bool {
        self.merged
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &VeRAConfig {
        &self.config
    }

    /// Dimension of the layer `(out_features, in_features)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.out_features, self.in_features)
    }
}

// ──────────────────────────── LCG helpers ────────────────────────────────────

/// Linear-congruential generator step (Knuth multiplicative hash).
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// Map a raw LCG state to a float in `[−1, 1)`.
#[inline]
fn lcg_to_uniform(state: u64) -> f64 {
    // Use upper 53 bits for mantissa → uniform in [0, 1), map to [-1, 1).
    let frac = (state >> 11) as f64 / (1u64 << 53) as f64;
    frac * 2.0 - 1.0
}

/// Map a raw LCG state to a float in `(0, 1)` (strictly open — safe for ln()).
#[inline]
fn lcg_to_unit(state: u64) -> f64 {
    let frac = (state >> 11) as f64 / (1u64 << 53) as f64;
    // Clamp away from 0 so Box–Muller ln() never diverges.
    if frac == 0.0 {
        f64::EPSILON
    } else {
        frac
    }
}

// ──────────────────────────── tests ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    // ── VeraLayer tests ──────────────────────────────────────────────────────

    fn make_vera_layer(out: usize, in_f: usize, rank: usize) -> VeraLayer {
        let w = Array2::from_shape_fn((out, in_f), |(i, j)| (i * in_f + j) as f64 * 0.1);
        VeraLayer::new(
            w,
            &VeraConfig {
                rank,
                ..Default::default()
            },
        )
        .expect("VeraLayer::new failed")
    }

    #[test]
    fn vera_delta_shape() {
        let layer = make_vera_layer(8, 6, 4);
        let delta = layer.compute_delta();
        assert_eq!(delta.shape(), &[8, 6]);
    }

    #[test]
    fn vera_zero_b_scale() {
        let mut layer = make_vera_layer(4, 6, 8);
        layer.b_scale.fill(0.0);
        let delta = layer.compute_delta();
        for v in delta.iter() {
            assert!(v.abs() < 1e-14, "expected zero delta, got {v}");
        }
    }

    #[test]
    fn vera_zero_d_scale() {
        let mut layer = make_vera_layer(4, 6, 8);
        layer.d_scale.fill(0.0);
        let delta = layer.compute_delta();
        for v in delta.iter() {
            assert!(v.abs() < 1e-14, "expected zero delta, got {v}");
        }
    }

    #[test]
    fn vera_reproducible() {
        // Two layers with identical config must produce identical A and B.
        let w = Array2::<f64>::eye(6);
        let cfg = VeraConfig {
            rank: 4,
            projection_seed: 1234,
            ..Default::default()
        };
        let l1 = VeraLayer::new(w.clone(), &cfg).expect("l1");
        let l2 = VeraLayer::new(w, &cfg).expect("l2");

        let a1 = l1.generate_a();
        let a2 = l2.generate_a();
        let b1 = l1.generate_b();
        let b2 = l2.generate_b();

        for (x, y) in a1.iter().zip(a2.iter()) {
            assert!((x - y).abs() < 1e-15, "A matrices differ");
        }
        for (x, y) in b1.iter().zip(b2.iter()) {
            assert!((x - y).abs() < 1e-15, "B matrices differ");
        }
    }

    #[test]
    fn vera_forward_output_shape() {
        let layer = make_vera_layer(5, 7, 16);
        let input = Array2::from_elem((3, 7), 0.5);
        let out = layer.forward(&input).expect("forward");
        assert_eq!(out.shape(), &[3, 5]);
    }

    #[test]
    fn vera_n_params_efficient() {
        let rank = 16_usize;
        let out = 64_usize;
        let in_f = 64_usize;
        let layer = make_vera_layer(out, in_f, rank);
        // VeRA trainable = rank + out (much less than rank*(in+out) for LoRA)
        let expected = rank + out;
        assert_eq!(layer.n_trainable_params(), expected);

        let lora_params = rank * in_f + out * rank; // 2048
        assert!(
            layer.n_trainable_params() < lora_params,
            "VeRA ({}) should use fewer params than LoRA ({})",
            layer.n_trainable_params(),
            lora_params
        );
    }

    #[test]
    fn vera_effective_weight_changes_with_scale() {
        let layer = make_vera_layer(4, 6, 8);
        let base = layer.effective_weight();

        let mut layer2 = make_vera_layer(4, 6, 8);
        layer2.b_scale.fill(2.0);
        let modified = layer2.effective_weight();

        let all_same = base
            .iter()
            .zip(modified.iter())
            .all(|(a, b)| (a - b).abs() < 1e-14);
        assert!(
            !all_same,
            "effective_weight did not change when b_scale changed"
        );
    }

    #[test]
    fn vera_invalid_rank_zero() {
        let w = Array2::<f64>::eye(4);
        let cfg = VeraConfig {
            rank: 0,
            ..Default::default()
        };
        assert!(VeraLayer::new(w, &cfg).is_err());
    }

    // ── VeRAConfig tests ─────────────────────────────────────────────────────

    #[test]
    fn vera_config_defaults() {
        let cfg = VeRAConfig::default();
        assert_eq!(cfg.rank, 8);
        assert!((cfg.alpha - 16.0).abs() < 1e-15);
        assert_eq!(cfg.seed, 42);
    }

    // ── SharedRandomMatrices tests ───────────────────────────────────────────

    #[test]
    fn shared_matrices_dimensions() {
        let shared = SharedRandomMatrices::new(32, 64, 8, 42);
        assert_eq!(shared.a.shape(), &[8, 32]);
        assert_eq!(shared.b.shape(), &[64, 8]);
    }

    #[test]
    fn shared_matrices_b_is_zeros() {
        let shared = SharedRandomMatrices::new(16, 32, 4, 7);
        for v in shared.b.iter() {
            assert_eq!(*v, 0.0, "shared.b must be zero-initialised");
        }
    }

    #[test]
    fn shared_matrices_a_nonzero() {
        let shared = SharedRandomMatrices::new(16, 32, 4, 7);
        let nonzero = shared.a.iter().any(|v| v.abs() > 1e-10);
        assert!(nonzero, "shared.a should have nonzero entries");
    }

    #[test]
    fn shared_matrices_reproducible() {
        let s1 = SharedRandomMatrices::new(10, 20, 4, 999);
        let s2 = SharedRandomMatrices::new(10, 20, 4, 999);
        for (x, y) in s1.a.iter().zip(s2.a.iter()) {
            assert!(
                (x - y).abs() < 1e-15,
                "SharedRandomMatrices not reproducible"
            );
        }
    }

    // ── VeRALayer helper ─────────────────────────────────────────────────────

    fn make_vera(out: usize, in_f: usize, rank: usize) -> VeRALayer {
        let w = Array2::from_shape_fn((out, in_f), |(i, j)| (i * in_f + j) as f64 * 0.05);
        let shared = SharedRandomMatrices::new(in_f, out, rank, 42);
        let cfg = VeRAConfig {
            rank,
            alpha: rank as f64,
            seed: 42,
        };
        VeRALayer::new(w, shared, &cfg).expect("VeRALayer::new failed")
    }

    // ── VeRALayer creation tests ─────────────────────────────────────────────

    #[test]
    fn vera_layer_creation_ok() {
        let _layer = make_vera(8, 16, 4);
    }

    #[test]
    fn vera_layer_dim_mismatch_a_rows() {
        let w = Array2::<f64>::zeros((8, 16));
        // shared.a has 3 rows but rank=4 → mismatch
        let mut shared = SharedRandomMatrices::new(16, 8, 4, 1);
        shared.a = Array2::zeros((3, 16)); // wrong rank
        let cfg = VeRAConfig {
            rank: 4,
            alpha: 4.0,
            seed: 1,
        };
        assert!(VeRALayer::new(w, shared, &cfg).is_err());
    }

    #[test]
    fn vera_layer_dim_mismatch_a_cols() {
        let w = Array2::<f64>::zeros((8, 16));
        // shared.a has wrong in_features
        let mut shared = SharedRandomMatrices::new(16, 8, 4, 1);
        shared.a = Array2::zeros((4, 10)); // cols 10 != in_features 16
        let cfg = VeRAConfig {
            rank: 4,
            alpha: 4.0,
            seed: 1,
        };
        assert!(VeRALayer::new(w, shared, &cfg).is_err());
    }

    #[test]
    fn vera_layer_dim_mismatch_b_rows() {
        let w = Array2::<f64>::zeros((8, 16));
        let mut shared = SharedRandomMatrices::new(16, 8, 4, 1);
        shared.b = Array2::zeros((5, 4)); // rows 5 != out_features 8
        let cfg = VeRAConfig {
            rank: 4,
            alpha: 4.0,
            seed: 1,
        };
        assert!(VeRALayer::new(w, shared, &cfg).is_err());
    }

    // ── delta_weight / effective_weight tests ────────────────────────────────

    #[test]
    fn vera_layer_delta_shape() {
        let layer = make_vera(6, 10, 4);
        let delta = layer.delta_weight();
        assert_eq!(delta.shape(), &[6, 10]);
    }

    #[test]
    fn vera_layer_effective_weight_shape() {
        let layer = make_vera(6, 10, 4);
        let eff = layer.effective_weight();
        assert_eq!(eff.shape(), &[6, 10]);
    }

    #[test]
    fn vera_layer_effective_weight_equals_weight_plus_delta() {
        let layer = make_vera(5, 8, 4);
        let delta = layer.delta_weight();
        let eff = layer.effective_weight();
        let expected = &layer.weight + &delta;
        for (a, b) in eff.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-14,
                "effective_weight != weight + delta_weight"
            );
        }
    }

    // ── trainable_params / total_params ──────────────────────────────────────

    #[test]
    fn vera_layer_trainable_params() {
        let rank = 4_usize;
        let out = 12_usize;
        let in_f = 20_usize;
        let layer = make_vera(out, in_f, rank);
        assert_eq!(layer.trainable_params(), rank + out);
    }

    #[test]
    fn vera_layer_total_params() {
        let out = 12_usize;
        let in_f = 20_usize;
        let layer = make_vera(out, in_f, 4);
        assert_eq!(layer.total_params(), out * in_f);
    }

    // ── merge / unmerge ──────────────────────────────────────────────────────

    #[test]
    fn vera_layer_merge_sets_flag() {
        let mut layer = make_vera(4, 8, 2);
        assert!(!layer.is_merged());
        layer.merge().expect("merge");
        assert!(layer.is_merged());
    }

    #[test]
    fn vera_layer_double_merge_errors() {
        let mut layer = make_vera(4, 8, 2);
        layer.merge().expect("first merge");
        assert!(layer.merge().is_err(), "second merge should fail");
    }

    #[test]
    fn vera_layer_unmerge_restores_flag() {
        let mut layer = make_vera(4, 8, 2);
        layer.merge().expect("merge");
        layer.unmerge().expect("unmerge");
        assert!(!layer.is_merged());
    }

    #[test]
    fn vera_layer_unmerge_without_merge_errors() {
        let mut layer = make_vera(4, 8, 2);
        assert!(
            layer.unmerge().is_err(),
            "unmerge without prior merge should fail"
        );
    }

    #[test]
    fn vera_layer_merge_unmerge_roundtrip() {
        let layer = make_vera(6, 12, 3);
        let eff_before = layer.effective_weight();

        let mut merged = make_vera(6, 12, 3);
        // Ensure same d/b_vec so merge/unmerge is mathematically exact.
        merged.d = layer.d.clone();
        merged.b_vec = layer.b_vec.clone();

        merged.merge().expect("merge");
        merged.unmerge().expect("unmerge");
        let eff_after = merged.effective_weight();

        for (a, b) in eff_before.iter().zip(eff_after.iter()) {
            assert!((a - b).abs() < 1e-12, "roundtrip error: {a} vs {b}");
        }
    }

    // ── forward ──────────────────────────────────────────────────────────────

    #[test]
    fn vera_layer_forward_shape() {
        let layer = make_vera(8, 16, 4);
        let input = Array2::from_elem((5, 16), 0.1_f64);
        let out = layer.forward(&input).expect("forward");
        assert_eq!(out.shape(), &[5, 8]);
    }

    #[test]
    fn vera_layer_forward_wrong_features_errors() {
        let layer = make_vera(8, 16, 4);
        let input = Array2::from_elem((3, 10), 0.1_f64); // wrong in_features
        assert!(layer.forward(&input).is_err());
    }

    // ── scaling ──────────────────────────────────────────────────────────────

    #[test]
    fn vera_layer_scaling_alpha_over_rank() {
        let cfg = VeRAConfig {
            rank: 4,
            alpha: 8.0,
            seed: 1,
        };
        let w = Array2::<f64>::eye(4);
        let shared = SharedRandomMatrices::new(4, 4, 4, 1);
        let layer = VeRALayer::new(w, shared, &cfg).expect("new");
        assert!((layer.scaling() - 2.0).abs() < 1e-15);
    }

    // ── shared-matrix reuse ───────────────────────────────────────────────────

    #[test]
    fn vera_layer_shared_matrices_reuse() {
        // Two layers sharing the same SharedRandomMatrices (cloned) must produce
        // deltas that derive from the same frozen projections.
        // in_features=8, out_features=6, rank=4
        let shared = SharedRandomMatrices::new(8, 6, 4, 77);
        let cfg = VeRAConfig {
            rank: 4,
            alpha: 4.0,
            seed: 77,
        };

        // Both layers have weight [out=6 × in=8].
        let w1 = Array2::<f64>::zeros((6, 8));
        let w2 = Array2::from_elem((6, 8), 0.1_f64);

        let mut layer1 = VeRALayer::new(w1, shared.clone(), &cfg).expect("l1");
        let mut layer2 = VeRALayer::new(w2, shared, &cfg).expect("l2");

        // Give both identical scaling vectors so deltas should be equal.
        let d_val = Array1::from_elem(4, 0.5_f64);
        let b_val = Array1::from_elem(6, 0.3_f64);
        layer1.d = d_val.clone();
        layer1.b_vec = b_val.clone();
        layer2.d = d_val;
        layer2.b_vec = b_val;

        let delta1 = layer1.delta_weight();
        let delta2 = layer2.delta_weight();

        for (a, b) in delta1.iter().zip(delta2.iter()) {
            assert!(
                (a - b).abs() < 1e-13,
                "deltas differ despite identical shared matrices: {a} vs {b}"
            );
        }
    }
}
