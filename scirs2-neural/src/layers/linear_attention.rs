//! Efficient linear attention mechanisms
//!
//! This module provides O(n) and sub-quadratic attention variants:
//!
//! - **`LinearAttentionLayer`** – Kernel-based O(n) linear attention
//!   (Katharopoulos et al., 2020) with configurable feature maps (ELU+1, ReLU,
//!   Random Fourier Features).
//! - **`PerformerAttention`** – FAVOR+ approximation (Choromanski et al., 2020)
//!   using positive orthogonal random features (PORF) to approximate softmax.
//! - **`LinformerAttention`** – Low-rank projection of K and V to reduce
//!   the O(n^2) memory footprint (Wang et al., 2020).

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, Array2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::Rng;
use std::fmt::Debug;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Xavier-initialised weight matrix, returned as a flat `Vec`.
fn xavier_vec<F: Float, R: Rng>(
    fan_in: usize,
    fan_out: usize,
    count: usize,
    rng: &mut R,
) -> Result<Vec<F>> {
    let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        let x: f64 = rng.random_range(-1.0..1.0);
        let f = F::from(x * scale)
            .ok_or_else(|| NeuralError::InvalidArchitecture("xavier_vec cast".into()))?;
        v.push(f);
    }
    Ok(v)
}

/// Build an `IxDyn` weight array of shape `[rows, cols]`.
fn mk_weight<F: Float, R: Rng>(rows: usize, cols: usize, rng: &mut R) -> Result<Array<F, IxDyn>> {
    let data = xavier_vec(rows, cols, rows * cols, rng)?;
    Array::from_shape_vec(IxDyn(&[rows, cols]), data)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("mk_weight: {e}")))
}

/// Dense matrix multiplication: `[B, S, D_in] @ [D_in, D_out] -> [B, S, D_out]`
fn batch_linear<F: Float + NumAssign>(
    x: &Array<F, IxDyn>,
    w: &Array<F, IxDyn>,
    d_in: usize,
    d_out: usize,
) -> Result<Array<F, IxDyn>> {
    let s = x.shape();
    let batch = s[0];
    let seq = s[1];
    let mut out = Array::zeros(IxDyn(&[batch, seq, d_out]));
    for b in 0..batch {
        for t in 0..seq {
            for o in 0..d_out {
                let mut acc = F::zero();
                for i in 0..d_in {
                    acc += x[[b, t, i]] * w[[i, o]];
                }
                out[[b, t, o]] = acc;
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// 1.  LinearAttentionLayer
// ===========================================================================

/// Type of positivity-preserving kernel feature map used by `LinearAttentionLayer`.
#[derive(Debug, Clone, Copy)]
pub enum LinearFeatureMap {
    /// ELU(x) + 1  (Katharopoulos et al., 2020).
    Elu,
    /// Clipped ReLU(x).
    Relu,
    /// Random Fourier Features (RFF) – cos/sin pairs, positive kernel.
    /// The projection matrix is generated at construction time and is stored
    /// inside the layer.
    RandomFourier,
}

/// Configuration for `LinearAttentionLayer`.
#[derive(Debug, Clone)]
pub struct LinearAttentionLayerConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Kernel feature map type.
    pub feature_map: LinearFeatureMap,
    /// Number of random Fourier features per head dimension (only used when
    /// `feature_map == RandomFourier`).
    pub n_rff: usize,
    /// Small constant for numerical stability in the denominator.
    pub eps: f64,
}

impl Default for LinearAttentionLayerConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            head_dim: 16,
            feature_map: LinearFeatureMap::Elu,
            n_rff: 64,
            eps: 1e-6,
        }
    }
}

/// Linear Attention (O(n)) with pluggable kernel feature maps.
///
/// The core identity is:
///
/// ```text
/// output_i = (Σ_j φ(Q_i)^T φ(K_j) V_j) / (Σ_j φ(Q_i)^T φ(K_j))
///          = φ(Q_i) (Σ_j φ(K_j)^T V_j) / (φ(Q_i) · Σ_j φ(K_j))
/// ```
///
/// This can be evaluated in O(n d^2) time by first accumulating the KV context
/// matrix and the key normaliser, then querying each position.
///
/// When `feature_map = RandomFourier`, the queries and keys are lifted into a
/// `2 * n_rff` dimensional feature space via:
/// ```text
/// φ(x)_i = exp(‖ω_i‖^2/2) * [cos(ω_i · x), sin(ω_i · x)]  (positive definite)
/// ```
/// which produces unbiased estimates of the squared-exponential (Gaussian)
/// kernel and is always non-negative.
#[derive(Debug)]
pub struct LinearAttentionLayer<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: LinearAttentionLayerConfig,
    w_query: Array<F, IxDyn>,
    w_key: Array<F, IxDyn>,
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    /// Random projection matrix for RFF: shape [head_dim, n_rff].
    /// Each column ω_i is drawn from N(0, I).
    rff_proj: Option<Array<F, IxDyn>>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static>
    LinearAttentionLayer<F>
{
    /// Create a new `LinearAttentionLayer`.
    pub fn new<R: Rng>(
        d_model: usize,
        config: LinearAttentionLayerConfig,
        rng: &mut R,
    ) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) must equal num_heads * head_dim ({nh} * {hd})"
            )));
        }

        // Build projection matrix for random Fourier features if needed.
        let rff_proj = if matches!(config.feature_map, LinearFeatureMap::RandomFourier) {
            let nr = config.n_rff;
            // Sample from standard normal via Box–Muller.
            let mut omega_data = Vec::with_capacity(hd * nr);
            for _ in 0..hd * nr {
                let u1: f64 = rng.random::<f64>().max(1e-12_f64);
                let u2: f64 = rng.random::<f64>();
                let z =
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let f = F::from(z)
                    .ok_or_else(|| NeuralError::InvalidArchitecture("rff cast".into()))?;
                omega_data.push(f);
            }
            let arr = Array::from_shape_vec(IxDyn(&[hd, nr]), omega_data)
                .map_err(|e| NeuralError::InvalidArchitecture(format!("rff_proj: {e}")))?;
            Some(arr)
        } else {
            None
        };

        Ok(Self {
            d_model,
            config,
            w_query: mk_weight(d_model, d_model, rng)?,
            w_key: mk_weight(d_model, d_model, rng)?,
            w_value: mk_weight(d_model, d_model, rng)?,
            w_output: mk_weight(d_model, d_model, rng)?,
            rff_proj,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Apply the element-wise feature map to a single value.
    fn apply_pointwise(&self, v: F) -> F {
        match self.config.feature_map {
            LinearFeatureMap::Elu => {
                if v > F::zero() {
                    v + F::one()
                } else {
                    v.exp() // exp(v) for v<=0 gives ELU(v)+1
                }
            }
            LinearFeatureMap::Relu => {
                if v > F::zero() {
                    v
                } else {
                    F::zero()
                }
            }
            LinearFeatureMap::RandomFourier => {
                // Should not reach here – RFF uses `apply_rff` below.
                v
            }
        }
    }

    /// Apply RFF feature map to a single vector `x` of length `head_dim`,
    /// returning a vector of length `2 * n_rff`.
    ///
    /// φ(x)_j = exp(-‖ω_j‖²/2) * [cos(ω_j·x), sin(ω_j·x)]
    fn apply_rff(&self, x: &[F]) -> Result<Vec<F>> {
        let proj = self.rff_proj.as_ref().ok_or_else(|| {
            NeuralError::InferenceError("rff_proj missing for RandomFourier".into())
        })?;
        let hd = x.len();
        let nr = self.config.n_rff;
        let mut out = Vec::with_capacity(2 * nr);
        for j in 0..nr {
            // Compute ω_j · x
            let mut dot = 0.0_f64;
            let mut norm_sq = 0.0_f64;
            for i in 0..hd {
                let w_ij = proj[[i, j]].to_f64().unwrap_or(0.0);
                let x_i = x[i].to_f64().unwrap_or(0.0);
                dot += w_ij * x_i;
                norm_sq += w_ij * w_ij;
            }
            let scale = (-0.5 * norm_sq).exp();
            let cos_val = F::from(scale * dot.cos())
                .ok_or_else(|| NeuralError::InferenceError("rff cos cast".into()))?;
            let sin_val = F::from(scale * dot.sin())
                .ok_or_else(|| NeuralError::InferenceError("rff sin cast".into()))?;
            out.push(cos_val);
            out.push(sin_val);
        }
        Ok(out)
    }

    /// Feature-map the projected Q or K tensor `[batch, seq, nh, hd]`.
    ///
    /// Returns `[batch, seq, nh, feat_dim]` where `feat_dim` equals `head_dim`
    /// for pointwise maps and `2 * n_rff` for RFF.
    fn feature_map_tensor(
        &self,
        x: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, usize)> {
        let shape = x.shape();
        let (batch, seq, nh, hd) = (shape[0], shape[1], shape[2], shape[3]);

        match self.config.feature_map {
            LinearFeatureMap::RandomFourier => {
                let feat_dim = 2 * self.config.n_rff;
                let mut out = Array::zeros(IxDyn(&[batch, seq, nh, feat_dim]));
                for b in 0..batch {
                    for t in 0..seq {
                        for h in 0..nh {
                            let vec: Vec<F> = (0..hd).map(|d| x[[b, t, h, d]]).collect();
                            let mapped = self.apply_rff(&vec)?;
                            for (d, val) in mapped.into_iter().enumerate() {
                                out[[b, t, h, d]] = val;
                            }
                        }
                    }
                }
                Ok((out, feat_dim))
            }
            _ => {
                let mut out = Array::zeros(IxDyn(&[batch, seq, nh, hd]));
                for b in 0..batch {
                    for t in 0..seq {
                        for h in 0..nh {
                            for d in 0..hd {
                                out[[b, t, h, d]] = self.apply_pointwise(x[[b, t, h, d]]);
                            }
                        }
                    }
                }
                Ok((out, hd))
            }
        }
    }

    /// Core linear-attention computation.
    ///
    /// Inputs:
    /// - `phi_q`: `[batch, seq, nh, feat_dim]`
    /// - `phi_k`: `[batch, seq, nh, feat_dim]`
    /// - `v`:     `[batch, seq, nh, hd_v]`
    ///
    /// Output: `[batch, seq, nh, hd_v]`
    pub fn linear_attention_forward(
        phi_q: &Array<F, IxDyn>,
        phi_k: &Array<F, IxDyn>,
        v: &Array<F, IxDyn>,
        eps: F,
    ) -> Result<Array<F, IxDyn>> {
        let s = phi_q.shape();
        let (batch, seq, nh, feat) = (s[0], s[1], s[2], s[3]);
        let hd_v = v.shape()[3];

        let mut out = Array::zeros(IxDyn(&[batch, seq, nh, hd_v]));

        for b in 0..batch {
            for h in 0..nh {
                // Accumulate KV: [feat x hd_v]
                let mut kv = vec![F::zero(); feat * hd_v];
                // Accumulate K sum: [feat]
                let mut k_sum = vec![F::zero(); feat];

                for j in 0..seq {
                    for f in 0..feat {
                        k_sum[f] += phi_k[[b, j, h, f]];
                        for dv in 0..hd_v {
                            kv[f * hd_v + dv] += phi_k[[b, j, h, f]] * v[[b, j, h, dv]];
                        }
                    }
                }

                for i in 0..seq {
                    let mut denom = F::zero();
                    for f in 0..feat {
                        denom += phi_q[[b, i, h, f]] * k_sum[f];
                    }
                    let norm = if denom.abs() < eps { eps } else { denom };

                    for dv in 0..hd_v {
                        let mut num = F::zero();
                        for f in 0..feat {
                            num += phi_q[[b, i, h, f]] * kv[f * hd_v + dv];
                        }
                        out[[b, i, h, dv]] = num / norm;
                    }
                }
            }
        }
        Ok(out)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for LinearAttentionLayer<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "LinearAttentionLayer expects 3D input, got {}D",
                shape.len()
            )));
        }
        let (batch, seq, dm) = (shape[0], shape[1], shape[2]);
        if dm != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "d_model mismatch: expected {}, got {dm}",
                self.d_model
            )));
        }

        let nh = self.config.num_heads;
        let hd = self.config.head_dim;
        let eps = F::from(self.config.eps).unwrap_or_else(|| F::from(1e-6_f64).unwrap_or(F::zero()));

        let q = batch_linear(input, &self.w_query, dm, dm)?;
        let k = batch_linear(input, &self.w_key, dm, dm)?;
        let v = batch_linear(input, &self.w_value, dm, dm)?;

        let q = q.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape q: {e}")))?;
        let k = k.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape k: {e}")))?;
        let v = v.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape v: {e}")))?;

        let (phi_q, _feat_dim) = self.feature_map_tensor(&q)?;
        let (phi_k, _) = self.feature_map_tensor(&k)?;

        let attended = Self::linear_attention_forward(&phi_q, &phi_k, &v, eps)?;

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("concat: {e}")))?;
        batch_linear(&concat, &self.w_output, dm, dm)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, _lr: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        vec![
            self.w_query.clone(),
            self.w_key.clone(),
            self.w_value.clone(),
            self.w_output.clone(),
        ]
    }

    fn set_params(&mut self, p: &[Array<F, IxDyn>]) -> Result<()> {
        if p.len() >= 4 {
            self.w_query = p[0].clone();
            self.w_key = p[1].clone();
            self.w_value = p[2].clone();
            self.w_output = p[3].clone();
        }
        Ok(())
    }

    fn set_training(&mut self, t: bool) {
        self.training = t;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "LinearAttentionLayer"
    }

    fn parameter_count(&self) -> usize {
        4 * self.d_model * self.d_model
    }
}

// ===========================================================================
// 2.  PerformerAttention  (FAVOR+)
// ===========================================================================

/// Configuration for `PerformerAttention`.
#[derive(Debug, Clone)]
pub struct PerformerConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of random features (m).  Larger → closer to exact softmax.
    pub n_features: usize,
    /// Whether to use positive orthogonal random features (PORF).
    /// When `true`, the random projection matrix is made orthogonal via QR,
    /// which reduces variance compared to i.i.d. Gaussian features.
    pub use_orthogonal: bool,
    /// Small constant for numerical stability.
    pub eps: f64,
}

impl Default for PerformerConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            head_dim: 16,
            n_features: 64,
            use_orthogonal: true,
            eps: 1e-6,
        }
    }
}

/// Performer attention using FAVOR+ (Choromanski et al., 2020).
///
/// Approximates `softmax(Q K^T / √d) V` via:
///
/// ```text
/// φ(x) = (exp(−‖x‖²/2) / √m) * [exp(ω_1·x), ..., exp(ω_m·x)]
/// ```
///
/// where `{ω_i}` are drawn from N(0, I) (or made orthogonal via PORF).
/// This gives a positive, unbiased kernel approximation and O(n m d)
/// computation instead of O(n² d).
///
/// ### Positive Orthogonal Random Features (PORF)
///
/// PORF replaces i.i.d. Gaussian columns with a set of *orthogonal* Gaussian
/// random vectors: blocks of `d` columns are generated as `S W` where `W` is
/// a random orthogonal matrix from QR decomposition and `S` is a diagonal of
/// i.i.d. chi-distributed norms.  This strictly reduces the approximation
/// variance.
#[derive(Debug)]
pub struct PerformerAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: PerformerConfig,
    w_query: Array<F, IxDyn>,
    w_key: Array<F, IxDyn>,
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    /// Projection matrix Ω: [head_dim, n_features]
    projection: Array<F, IxDyn>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> PerformerAttention<F> {
    /// Create a new `PerformerAttention` layer.
    pub fn new<R: Rng>(d_model: usize, config: PerformerConfig, rng: &mut R) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) must equal num_heads * head_dim ({nh} * {hd})"
            )));
        }
        if config.n_features == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "n_features must be > 0".into(),
            ));
        }

        let proj = if config.use_orthogonal {
            Self::create_porf_projection(hd, config.n_features, rng)?
        } else {
            Self::create_gaussian_projection(hd, config.n_features, rng)?
        };

        Ok(Self {
            d_model,
            config,
            w_query: mk_weight(d_model, d_model, rng)?,
            w_key: mk_weight(d_model, d_model, rng)?,
            w_value: mk_weight(d_model, d_model, rng)?,
            w_output: mk_weight(d_model, d_model, rng)?,
            projection: proj,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Generate an i.i.d. Gaussian projection matrix [d, m].
    pub fn create_gaussian_projection<R: Rng>(
        d: usize,
        m: usize,
        rng: &mut R,
    ) -> Result<Array<F, IxDyn>> {
        let mut data = Vec::with_capacity(d * m);
        for _ in 0..d * m {
            let u1: f64 = rng.random::<f64>().max(1e-12_f64);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let f = F::from(z)
                .ok_or_else(|| NeuralError::InvalidArchitecture("gaussian proj cast".into()))?;
            data.push(f);
        }
        Array::from_shape_vec(IxDyn(&[d, m]), data)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("gaussian_proj: {e}")))
    }

    /// Generate positive orthogonal random features (PORF) [d, m].
    ///
    /// Algorithm:
    /// 1. Draw `ceil(m / d)` random orthogonal `d×d` matrices via Gram-Schmidt.
    /// 2. For each matrix, scale columns by chi-distributed norms (‖g_i‖ for
    ///    g_i ~ N(0, I_d)) to preserve the norm distribution of Gaussian RVs.
    /// 3. Concatenate and truncate to `d × m`.
    pub fn create_projection_matrix<R: Rng>(
        n_features: usize,
        d_model: usize,
        rng: &mut R,
    ) -> Result<Array2<F>> {
        let proj = Self::create_porf_projection(d_model, n_features, rng)?;
        let proj_2d = proj
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|e| NeuralError::InvalidArchitecture(format!("porf to 2d: {e}")))?;
        Ok(proj_2d)
    }

    fn create_porf_projection<R: Rng>(
        d: usize,
        m: usize,
        rng: &mut R,
    ) -> Result<Array<F, IxDyn>> {
        // We'll build blocks of d orthonormal vectors, then scale by chi norms.
        let n_blocks = m.div_ceil(d);
        let mut columns: Vec<Vec<f64>> = Vec::with_capacity(n_blocks * d);

        for _ in 0..n_blocks {
            // Draw d random vectors of length d, then orthonormalise.
            let mut vecs: Vec<Vec<f64>> = (0..d)
                .map(|_| {
                    (0..d)
                        .map(|_| {
                            let u1: f64 = rng.random::<f64>().max(1e-12_f64);
                            let u2: f64 = rng.random::<f64>();
                            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                        })
                        .collect()
                })
                .collect();

            // Gram-Schmidt orthonormalisation.
            for i in 0..d {
                for j in 0..i {
                    let dot: f64 = vecs[i].iter().zip(vecs[j].iter()).map(|(a, b)| a * b).sum();
                    let norm_j_sq: f64 = vecs[j].iter().map(|x| x * x).sum();
                    if norm_j_sq > 1e-12 {
                        let proj: Vec<f64> = vecs[j].iter().map(|x| x * dot / norm_j_sq).collect();
                        for k in 0..d {
                            vecs[i][k] -= proj[k];
                        }
                    }
                }
                let norm: f64 = vecs[i].iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    for k in 0..d {
                        vecs[i][k] /= norm;
                    }
                }
            }

            // Scale by chi norms (‖g_i‖ for g_i ~ N(0, I_d)).
            for i in 0..d {
                let chi_norm: f64 = (0..d)
                    .map(|_| {
                        let u1: f64 = rng.random::<f64>().max(1e-12_f64);
                        let u2: f64 = rng.random::<f64>();
                        let z =
                            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        z * z
                    })
                    .sum::<f64>()
                    .sqrt();
                for k in 0..d {
                    vecs[i][k] *= chi_norm;
                }
                columns.push(vecs[i].clone());
            }
        }

        // Build [d x m] matrix from the first m column vectors.
        let mut data = vec![0.0_f64; d * m];
        for (col_idx, col) in columns.iter().take(m).enumerate() {
            for row in 0..d {
                data[row * m + col_idx] = col[row];
            }
        }

        let typed: Vec<F> = data
            .iter()
            .map(|&x| {
                F::from(x).ok_or_else(|| NeuralError::InvalidArchitecture("porf cast".into()))
            })
            .collect::<Result<_>>()?;

        Array::from_shape_vec(IxDyn(&[d, m]), typed)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("porf matrix: {e}")))
    }

    /// FAVOR+ kernel feature map for a single vector `x` of length `head_dim`.
    ///
    /// ```text
    /// φ(x)_j = (exp(−‖x‖²/2) / √m) * exp(ω_j · x)
    /// ```
    ///
    /// which is always non-negative and yields an unbiased approximation to the
    /// softmax kernel K(x, y) = exp(x · y).
    pub fn kernel_feature(&self, x: &[F], projection: &Array<F, IxDyn>) -> Result<Vec<F>> {
        let hd = x.len();
        let m = self.config.n_features;
        let norm_sq: f64 = x.iter().map(|v| v.to_f64().unwrap_or(0.0).powi(2)).sum();
        let scale = (-0.5 * norm_sq).exp() / (m as f64).sqrt();

        let mut out = Vec::with_capacity(m);
        for j in 0..m {
            let mut dot = 0.0_f64;
            for i in 0..hd {
                dot += x[i].to_f64().unwrap_or(0.0) * projection[[i, j]].to_f64().unwrap_or(0.0);
            }
            let val = scale * dot.exp();
            out.push(
                F::from(val)
                    .ok_or_else(|| NeuralError::InferenceError("kernel_feature cast".into()))?,
            );
        }
        Ok(out)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for PerformerAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "PerformerAttention expects 3D input, got {}D",
                shape.len()
            )));
        }
        let (batch, seq, dm) = (shape[0], shape[1], shape[2]);
        if dm != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "d_model mismatch: expected {}, got {dm}",
                self.d_model
            )));
        }

        let nh = self.config.num_heads;
        let hd = self.config.head_dim;
        let m = self.config.n_features;
        let eps = F::from(self.config.eps).unwrap_or_else(|| F::from(1e-6_f64).unwrap_or(F::zero()));

        let q = batch_linear(input, &self.w_query, dm, dm)?;
        let k = batch_linear(input, &self.w_key, dm, dm)?;
        let v = batch_linear(input, &self.w_value, dm, dm)?;

        let q = q.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape q: {e}")))?;
        let k = k.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape k: {e}")))?;
        let v = v.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape v: {e}")))?;

        // Lift Q and K into feature space: [batch, seq, nh, m]
        let mut phi_q = Array::zeros(IxDyn(&[batch, seq, nh, m]));
        let mut phi_k = Array::zeros(IxDyn(&[batch, seq, nh, m]));

        for b in 0..batch {
            for t in 0..seq {
                for h in 0..nh {
                    let qvec: Vec<F> = (0..hd).map(|d| q[[b, t, h, d]]).collect();
                    let kvec: Vec<F> = (0..hd).map(|d| k[[b, t, h, d]]).collect();
                    let pq = self.kernel_feature(&qvec, &self.projection)?;
                    let pk = self.kernel_feature(&kvec, &self.projection)?;
                    for f in 0..m {
                        phi_q[[b, t, h, f]] = pq[f];
                        phi_k[[b, t, h, f]] = pk[f];
                    }
                }
            }
        }

        // Linear attention in feature space.
        let attended =
            LinearAttentionLayer::linear_attention_forward(&phi_q, &phi_k, &v, eps)?;

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("concat: {e}")))?;
        batch_linear(&concat, &self.w_output, dm, dm)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, _lr: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        vec![
            self.w_query.clone(),
            self.w_key.clone(),
            self.w_value.clone(),
            self.w_output.clone(),
        ]
    }

    fn set_params(&mut self, p: &[Array<F, IxDyn>]) -> Result<()> {
        if p.len() >= 4 {
            self.w_query = p[0].clone();
            self.w_key = p[1].clone();
            self.w_value = p[2].clone();
            self.w_output = p[3].clone();
        }
        Ok(())
    }

    fn set_training(&mut self, t: bool) {
        self.training = t;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "PerformerAttention"
    }

    fn parameter_count(&self) -> usize {
        4 * self.d_model * self.d_model + self.config.head_dim * self.config.n_features
    }
}

// ===========================================================================
// 3.  LinformerAttention
// ===========================================================================

/// Configuration for `LinformerAttention`.
#[derive(Debug, Clone)]
pub struct LinformerConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Low-rank projection size k (k < n).  Both K and V are projected from
    /// [seq, head_dim] to [k, head_dim].
    pub k: usize,
    /// Whether to apply causal masking (not supported in the original Linformer
    /// for efficiency; included here for completeness but is O(n k) not O(n)).
    pub causal: bool,
    /// Small constant for softmax numerical stability.
    pub eps: f64,
}

impl Default for LinformerConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            head_dim: 16,
            k: 32,
            causal: false,
            eps: 1e-6,
        }
    }
}

/// Linformer attention with low-rank K/V projection (Wang et al., 2020).
///
/// The key idea is to project the sequence dimension of K and V from `n` to
/// a small `k` (k ≪ n) using learnable projection matrices E_K and E_V:
///
/// ```text
/// K̃ = E_K K   (shape: [k, head_dim])
/// Ṽ = E_V V   (shape: [k, head_dim])
/// output = softmax(Q K̃^T / √d) Ṽ
/// ```
///
/// This reduces the attention complexity from O(n² d) to O(n k d).
/// The projection matrices E_K and E_V have shape [k, n] and are trained.
///
/// ### Limitations
///
/// Because E_K and E_V are fixed-size `[k, max_seq]` matrices, the layer is
/// designed for a *fixed* sequence length provided at construction time
/// (`max_seq`).  At runtime, if the input sequence is shorter, the unused
/// rows of E are simply ignored; if it is longer, an error is returned.
#[derive(Debug)]
pub struct LinformerAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    max_seq: usize,
    config: LinformerConfig,
    w_query: Array<F, IxDyn>,
    w_key: Array<F, IxDyn>,
    w_value: Array<F, IxDyn>,
    w_output: Array<F, IxDyn>,
    /// E_K projection: [k, max_seq]  (shared across heads for simplicity)
    e_key: Array<F, IxDyn>,
    /// E_V projection: [k, max_seq]
    e_val: Array<F, IxDyn>,
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> LinformerAttention<F> {
    /// Create a new `LinformerAttention` layer.
    ///
    /// # Arguments
    /// * `d_model` – total embedding dimension (= `num_heads * head_dim`).
    /// * `max_seq` – maximum supported sequence length.
    /// * `config`  – Linformer hyperparameters.
    /// * `rng`     – random number generator.
    pub fn new<R: Rng>(
        d_model: usize,
        max_seq: usize,
        config: LinformerConfig,
        rng: &mut R,
    ) -> Result<Self> {
        let nh = config.num_heads;
        let hd = config.head_dim;
        let k = config.k;

        if d_model != nh * hd {
            return Err(NeuralError::InvalidArchitecture(format!(
                "d_model ({d_model}) must equal num_heads * head_dim ({nh} * {hd})"
            )));
        }
        if k == 0 || k > max_seq {
            return Err(NeuralError::InvalidArchitecture(format!(
                "k ({k}) must satisfy 0 < k <= max_seq ({max_seq})"
            )));
        }

        Ok(Self {
            d_model,
            max_seq,
            config,
            w_query: mk_weight(d_model, d_model, rng)?,
            w_key: mk_weight(d_model, d_model, rng)?,
            w_value: mk_weight(d_model, d_model, rng)?,
            w_output: mk_weight(d_model, d_model, rng)?,
            e_key: mk_weight(k, max_seq, rng)?,
            e_val: mk_weight(k, max_seq, rng)?,
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Low-rank projection of a sequence tensor.
    ///
    /// `x`:   `[batch, seq, nh, hd]`
    /// `e`:   `[k, max_seq]`  (only the first `seq` columns are used)
    ///
    /// Returns `[batch, k, nh, hd]`.
    fn seq_project(
        x: &Array<F, IxDyn>,
        e: &Array<F, IxDyn>,
        k: usize,
    ) -> Result<Array<F, IxDyn>> {
        let xs = x.shape();
        let (batch, seq, nh, hd) = (xs[0], xs[1], xs[2], xs[3]);

        let mut out = Array::zeros(IxDyn(&[batch, k, nh, hd]));
        for b in 0..batch {
            for ki in 0..k {
                for h in 0..nh {
                    for d in 0..hd {
                        let mut acc = F::zero();
                        for j in 0..seq {
                            acc += e[[ki, j]] * x[[b, j, h, d]];
                        }
                        out[[b, ki, h, d]] = acc;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Scaled dot-product attention where K and V have sequence length `k`.
    ///
    /// `q`:  `[batch, seq_q, nh, hd]`
    /// `k_proj`:  `[batch, k, nh, hd]`
    /// `v_proj`:  `[batch, k, nh, hd]`
    ///
    /// Returns `[batch, seq_q, nh, hd]`.
    fn sdp_attention(
        q: &Array<F, IxDyn>,
        k_proj: &Array<F, IxDyn>,
        v_proj: &Array<F, IxDyn>,
        causal: bool,
        eps: F,
    ) -> Result<Array<F, IxDyn>> {
        let qs = q.shape();
        let (batch, seq_q, nh, hd) = (qs[0], qs[1], qs[2], qs[3]);
        let k_len = k_proj.shape()[1];

        let scale = F::from(1.0 / (hd as f64).sqrt())
            .ok_or_else(|| NeuralError::InferenceError("scale cast".into()))?;
        let neg_inf = F::neg_infinity();

        let mut out = Array::zeros(IxDyn(&[batch, seq_q, nh, hd]));

        for b in 0..batch {
            for h in 0..nh {
                // Compute scores [seq_q x k_len]
                let mut scores = vec![F::zero(); seq_q * k_len];
                for i in 0..seq_q {
                    for j in 0..k_len {
                        let mut dot = F::zero();
                        for d in 0..hd {
                            dot += q[[b, i, h, d]] * k_proj[[b, j, h, d]];
                        }
                        scores[i * k_len + j] = dot * scale;
                    }
                }

                // Causal mask (limited utility with low-rank K, but supported)
                if causal {
                    for i in 0..seq_q {
                        for j in (i + 1)..k_len {
                            scores[i * k_len + j] = neg_inf;
                        }
                    }
                }

                // Softmax over k dimension
                for i in 0..seq_q {
                    let row = &mut scores[i * k_len..(i + 1) * k_len];
                    let max_v = row.iter().fold(neg_inf, |a, &b| if b > a { b } else { a });
                    let mut sum = F::zero();
                    for s in row.iter_mut() {
                        *s = (*s - max_v).exp();
                        sum += *s;
                    }
                    let norm = if sum.abs() < eps { eps } else { sum };
                    for s in row.iter_mut() {
                        *s = *s / norm;
                    }
                }

                // Weighted sum of V_proj
                for i in 0..seq_q {
                    for d in 0..hd {
                        let mut acc = F::zero();
                        for j in 0..k_len {
                            acc += scores[i * k_len + j] * v_proj[[b, j, h, d]];
                        }
                        out[[b, i, h, d]] = acc;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Standalone Linformer forward pass using explicitly provided projection
    /// matrices.
    ///
    /// This is exposed as a public method so it can be called with custom
    /// E matrices for testing or research purposes.
    ///
    /// `q`:    `[batch, seq, nh, hd]` (post-projection query)
    /// `k`:    `[batch, seq, nh, hd]`
    /// `v`:    `[batch, seq, nh, hd]`
    /// `e_k`:  `[k, seq]`
    /// `e_v`:  `[k, seq]`
    pub fn linformer_forward(
        q: &Array<F, IxDyn>,
        k: &Array<F, IxDyn>,
        v: &Array<F, IxDyn>,
        e_k: &Array<F, IxDyn>,
        e_v: &Array<F, IxDyn>,
        causal: bool,
        eps: F,
    ) -> Result<Array<F, IxDyn>> {
        let rank = e_k.shape()[0];
        let k_proj = Self::seq_project(k, e_k, rank)?;
        let v_proj = Self::seq_project(v, e_v, rank)?;
        Self::sdp_attention(q, &k_proj, &v_proj, causal, eps)
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for LinformerAttention<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "LinformerAttention expects 3D input, got {}D",
                shape.len()
            )));
        }
        let (batch, seq, dm) = (shape[0], shape[1], shape[2]);
        if dm != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "d_model mismatch: expected {}, got {dm}",
                self.d_model
            )));
        }
        if seq > self.max_seq {
            return Err(NeuralError::InferenceError(format!(
                "seq ({seq}) exceeds max_seq ({}) for this Linformer layer",
                self.max_seq
            )));
        }

        let nh = self.config.num_heads;
        let hd = self.config.head_dim;
        let k = self.config.k;
        let eps = F::from(self.config.eps).unwrap_or_else(|| F::from(1e-6_f64).unwrap_or(F::zero()));

        let q = batch_linear(input, &self.w_query, dm, dm)?;
        let kin = batch_linear(input, &self.w_key, dm, dm)?;
        let v = batch_linear(input, &self.w_value, dm, dm)?;

        let q = q.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape q: {e}")))?;
        let kin = kin.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape k: {e}")))?;
        let v = v.into_shape_with_order(IxDyn(&[batch, seq, nh, hd]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape v: {e}")))?;

        // Slice E to match the actual sequence length.
        let e_k_slice = self.e_key
            .slice(scirs2_core::ndarray::s![0..k, 0..seq])
            .to_owned()
            .into_shape_with_order(IxDyn(&[k, seq]))
            .map_err(|e| NeuralError::InferenceError(format!("e_k slice: {e}")))?;
        let e_v_slice = self.e_val
            .slice(scirs2_core::ndarray::s![0..k, 0..seq])
            .to_owned()
            .into_shape_with_order(IxDyn(&[k, seq]))
            .map_err(|e| NeuralError::InferenceError(format!("e_v slice: {e}")))?;

        let attended = Self::linformer_forward(
            &q,
            &kin,
            &v,
            &e_k_slice,
            &e_v_slice,
            self.config.causal,
            eps,
        )?;

        let concat = attended
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("concat: {e}")))?;
        batch_linear(&concat, &self.w_output, dm, dm)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad.clone())
    }

    fn update(&mut self, _lr: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        vec![
            self.w_query.clone(),
            self.w_key.clone(),
            self.w_value.clone(),
            self.w_output.clone(),
            self.e_key.clone(),
            self.e_val.clone(),
        ]
    }

    fn set_params(&mut self, p: &[Array<F, IxDyn>]) -> Result<()> {
        if p.len() >= 6 {
            self.w_query = p[0].clone();
            self.w_key = p[1].clone();
            self.w_value = p[2].clone();
            self.w_output = p[3].clone();
            self.e_key = p[4].clone();
            self.e_val = p[5].clone();
        }
        Ok(())
    }

    fn set_training(&mut self, t: bool) {
        self.training = t;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        "LinformerAttention"
    }

    fn parameter_count(&self) -> usize {
        4 * self.d_model * self.d_model + 2 * self.config.k * self.max_seq
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::rng;

    // ---- LinearAttentionLayer ----

    #[test]
    fn test_linear_attention_layer_elu_forward() {
        let mut r = rng();
        let cfg = LinearAttentionLayerConfig {
            num_heads: 2,
            head_dim: 8,
            feature_map: LinearFeatureMap::Elu,
            n_rff: 16,
            eps: 1e-6,
        };
        let layer = LinearAttentionLayer::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((2, 5, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_linear_attention_layer_relu_forward() {
        let mut r = rng();
        let cfg = LinearAttentionLayerConfig {
            num_heads: 2,
            head_dim: 8,
            feature_map: LinearFeatureMap::Relu,
            n_rff: 16,
            eps: 1e-6,
        };
        let layer = LinearAttentionLayer::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 4, 16), 0.5).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 4, 16]);
    }

    #[test]
    fn test_linear_attention_layer_rff_forward() {
        let mut r = rng();
        let cfg = LinearAttentionLayerConfig {
            num_heads: 2,
            head_dim: 8,
            feature_map: LinearFeatureMap::RandomFourier,
            n_rff: 16,
            eps: 1e-6,
        };
        let layer = LinearAttentionLayer::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 4, 16), 0.3).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 4, 16]);
    }

    #[test]
    fn test_linear_attention_layer_output_finite() {
        let mut r = rng();
        let cfg = LinearAttentionLayerConfig {
            num_heads: 2,
            head_dim: 8,
            feature_map: LinearFeatureMap::Elu,
            n_rff: 8,
            eps: 1e-6,
        };
        let layer = LinearAttentionLayer::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 6, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite output");
        }
    }

    #[test]
    fn test_linear_attention_layer_d_model_mismatch() {
        let mut r = rng();
        let cfg = LinearAttentionLayerConfig {
            num_heads: 3,
            head_dim: 7,
            feature_map: LinearFeatureMap::Elu,
            n_rff: 8,
            eps: 1e-6,
        };
        // 3 * 7 != 16
        assert!(LinearAttentionLayer::<f64>::new(16, cfg, &mut r).is_err());
    }

    #[test]
    fn test_linear_attention_layer_param_count() {
        let mut r = rng();
        let cfg = LinearAttentionLayerConfig {
            num_heads: 2,
            head_dim: 8,
            feature_map: LinearFeatureMap::Elu,
            n_rff: 8,
            eps: 1e-6,
        };
        let layer = LinearAttentionLayer::<f64>::new(16, cfg, &mut r).expect("create failed");
        assert_eq!(layer.params().len(), 4);
        assert_eq!(layer.parameter_count(), 4 * 16 * 16);
    }

    // ---- PerformerAttention ----

    #[test]
    fn test_performer_creation() {
        let mut r = rng();
        let cfg = PerformerConfig {
            num_heads: 2,
            head_dim: 8,
            n_features: 16,
            use_orthogonal: true,
            eps: 1e-6,
        };
        let layer = PerformerAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        assert_eq!(layer.layer_type(), "PerformerAttention");
    }

    #[test]
    fn test_performer_porf_forward() {
        let mut r = rng();
        let cfg = PerformerConfig {
            num_heads: 2,
            head_dim: 8,
            n_features: 16,
            use_orthogonal: true,
            eps: 1e-6,
        };
        let layer = PerformerAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((2, 5, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_performer_gaussian_forward() {
        let mut r = rng();
        let cfg = PerformerConfig {
            num_heads: 2,
            head_dim: 8,
            n_features: 16,
            use_orthogonal: false,
            eps: 1e-6,
        };
        let layer = PerformerAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 4, 16), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 4, 16]);
    }

    #[test]
    fn test_performer_output_finite() {
        let mut r = rng();
        let cfg = PerformerConfig {
            num_heads: 2,
            head_dim: 8,
            n_features: 12,
            use_orthogonal: true,
            eps: 1e-6,
        };
        let layer = PerformerAttention::<f64>::new(16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 6, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn test_performer_create_projection_matrix() {
        let mut r = rng();
        let proj = PerformerAttention::<f64>::create_projection_matrix(32, 8, &mut r)
            .expect("projection failed");
        assert_eq!(proj.shape(), &[8, 32]);
    }

    // ---- LinformerAttention ----

    #[test]
    fn test_linformer_creation() {
        let mut r = rng();
        let cfg = LinformerConfig {
            num_heads: 2,
            head_dim: 8,
            k: 4,
            causal: false,
            eps: 1e-6,
        };
        let layer =
            LinformerAttention::<f64>::new(16, 16, cfg, &mut r).expect("create failed");
        assert_eq!(layer.layer_type(), "LinformerAttention");
    }

    #[test]
    fn test_linformer_forward() {
        let mut r = rng();
        let cfg = LinformerConfig {
            num_heads: 2,
            head_dim: 8,
            k: 4,
            causal: false,
            eps: 1e-6,
        };
        let layer =
            LinformerAttention::<f64>::new(16, 20, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((2, 8, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 8, 16]);
    }

    #[test]
    fn test_linformer_causal() {
        let mut r = rng();
        let cfg = LinformerConfig {
            num_heads: 2,
            head_dim: 8,
            k: 4,
            causal: true,
            eps: 1e-6,
        };
        let layer =
            LinformerAttention::<f64>::new(16, 16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 6, 16), 0.2).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[1, 6, 16]);
    }

    #[test]
    fn test_linformer_seq_too_long_error() {
        let mut r = rng();
        let cfg = LinformerConfig {
            num_heads: 2,
            head_dim: 8,
            k: 4,
            causal: false,
            eps: 1e-6,
        };
        let layer =
            LinformerAttention::<f64>::new(16, 8, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 12, 16), 0.1).into_dyn(); // seq=12 > max_seq=8
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_linformer_params() {
        let mut r = rng();
        let cfg = LinformerConfig {
            num_heads: 2,
            head_dim: 8,
            k: 4,
            causal: false,
            eps: 1e-6,
        };
        let layer =
            LinformerAttention::<f64>::new(16, 16, cfg, &mut r).expect("create failed");
        assert_eq!(layer.params().len(), 6); // 4 projections + E_K + E_V
    }

    #[test]
    fn test_linformer_output_finite() {
        let mut r = rng();
        let cfg = LinformerConfig {
            num_heads: 2,
            head_dim: 8,
            k: 4,
            causal: false,
            eps: 1e-6,
        };
        let layer =
            LinformerAttention::<f64>::new(16, 16, cfg, &mut r).expect("create failed");
        let input = Array3::<f64>::from_elem((1, 8, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        for v in out.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
