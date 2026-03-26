//! Graph Masked Autoencoder (GraphMAE, Hou et al. 2022).
//!
//! GraphMAE is a generative self-supervised learning method that:
//!
//! 1. **Masks** a random subset of node features by replacing them with a
//!    learnable mask token.
//! 2. **Encodes** the masked graph with a GNN encoder (here approximated by a
//!    linear layer + ReLU for simplicity).
//! 3. **Decodes** the latent representation back to the original feature space
//!    via a lightweight decoder.
//! 4. Computes the **scaled cosine error (SCE)** reconstruction loss only over
//!    the masked nodes:
//!
//! ```text
//! L_SCE = (1/|M|) Σ_{i∈M} (1 - cos_sim(ẑ_i, h_i)^γ)
//! ```
//!
//! where `ẑ_i` is the reconstructed feature and `h_i` is the original feature,
//! and `γ ≥ 1` controls the sharpness of the penalty.
//!
//! ## Reference
//! Hou, Z., Liu, X., Cen, Y., Dong, Y., Yang, H., Wang, C., & Tang, J. (2022).
//! *GraphMAE: Self-Supervised Masked Graph Autoencoders.* KDD 2022.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt, SeedableRng};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Graph Masked Autoencoder.
#[derive(Debug, Clone)]
pub struct GraphMaeConfig {
    /// Fraction of nodes whose features are masked during training.
    pub mask_rate: f64,
    /// Encoder output (latent) dimension.
    pub encoder_dim: usize,
    /// Decoder output dimension (must equal the input feature dimension).
    pub decoder_dim: usize,
    /// Scale of the random initialisation for the mask replacement token.
    pub replace_token_scale: f64,
}

impl Default for GraphMaeConfig {
    fn default() -> Self {
        Self {
            mask_rate: 0.25,
            encoder_dim: 64,
            decoder_dim: 64,
            replace_token_scale: 0.1,
        }
    }
}

// ============================================================================
// GraphMae
// ============================================================================

/// Graph Masked Autoencoder.
///
/// Maintains:
/// - A learnable **mask token** of shape `[feature_dim]` used to replace masked
///   node features.
/// - An **encoder weight** matrix `[feature_dim × encoder_dim]`.
/// - A **decoder weight** matrix `[encoder_dim × feature_dim]`.
pub struct GraphMae {
    /// Learnable mask replacement token: `[feature_dim]`
    mask_token: Array1<f64>,
    /// Encoder weight: `[feature_dim × encoder_dim]`
    encoder_weight: Array2<f64>,
    /// Decoder weight: `[encoder_dim × feature_dim]`
    decoder_weight: Array2<f64>,
    /// Feature dimension (= decoder output dimension).
    feature_dim: usize,
    config: GraphMaeConfig,
}

impl GraphMae {
    /// Construct a new GraphMAE.
    ///
    /// # Arguments
    /// * `feature_dim` – dimension of each node's input features
    /// * `config`      – MAE hyper-parameters
    /// * `seed`        – RNG seed for reproducible initialisation
    pub fn new(feature_dim: usize, config: GraphMaeConfig, seed: u64) -> Self {
        let mut rng = scirs2_core::random::ChaCha20Rng::seed_from_u64(seed);

        // Mask token: uniform in [-replace_token_scale, +replace_token_scale]
        let s = config.replace_token_scale;
        let mask_token = Array1::from_shape_fn(feature_dim, |_| rng.random::<f64>() * 2.0 * s - s);

        // Encoder: Xavier uniform [feature_dim × encoder_dim]
        let enc_scale = (6.0 / (feature_dim + config.encoder_dim) as f64).sqrt();
        let encoder_weight = Array2::from_shape_fn((feature_dim, config.encoder_dim), |_| {
            rng.random::<f64>() * 2.0 * enc_scale - enc_scale
        });

        // Decoder: Xavier uniform [encoder_dim × feature_dim]
        let dec_scale = (6.0 / (config.encoder_dim + feature_dim) as f64).sqrt();
        let decoder_weight = Array2::from_shape_fn((config.encoder_dim, feature_dim), |_| {
            rng.random::<f64>() * 2.0 * dec_scale - dec_scale
        });

        GraphMae {
            mask_token,
            encoder_weight,
            decoder_weight,
            feature_dim,
            config,
        }
    }

    /// Apply random feature masking.
    ///
    /// Selects a random subset of nodes (fraction ≈ `config.mask_rate`) and
    /// replaces their feature vectors with the learnable mask token.
    ///
    /// # Arguments
    /// * `features` – node feature matrix `[n_nodes × feature_dim]`
    /// * `seed`     – RNG seed (different from the model seed so each call can
    ///   produce a different mask)
    ///
    /// # Returns
    /// `(masked_features, mask_indices)` where `mask_indices` contains the
    /// row indices of the masked nodes (sorted ascending).
    pub fn mask_features(&self, features: &Array2<f64>, seed: u64) -> (Array2<f64>, Vec<usize>) {
        let n_nodes = features.dim().0;
        let mut rng = scirs2_core::random::ChaCha20Rng::seed_from_u64(seed);

        let mut masked = features.clone();
        let mut mask_indices = Vec::new();

        for i in 0..n_nodes {
            if rng.random::<f64>() < self.config.mask_rate {
                mask_indices.push(i);
                for d in 0..self.feature_dim {
                    masked[[i, d]] = self.mask_token[d];
                }
            }
        }

        mask_indices.sort_unstable();
        (masked, mask_indices)
    }

    /// Encode node features: `Z = ReLU(X @ W_enc)`
    ///
    /// # Arguments
    /// * `features` – (possibly masked) node features `[n_nodes × feature_dim]`
    ///
    /// # Returns
    /// Latent representations `[n_nodes × encoder_dim]`
    pub fn encode(&self, features: &Array2<f64>) -> Array2<f64> {
        let n_nodes = features.dim().0;
        let enc_dim = self.config.encoder_dim;

        let mut z = Array2::zeros((n_nodes, enc_dim));
        for i in 0..n_nodes {
            for k in 0..enc_dim {
                let mut val = 0.0;
                for d in 0..self.feature_dim {
                    val += features[[i, d]] * self.encoder_weight[[d, k]];
                }
                z[[i, k]] = if val > 0.0 { val } else { 0.0 }; // ReLU
            }
        }
        z
    }

    /// Decode latent representations: `X̂ = Z @ W_dec`
    ///
    /// # Arguments
    /// * `encoded` – latent representations `[n_nodes × encoder_dim]`
    ///
    /// # Returns
    /// Reconstructed features `[n_nodes × feature_dim]`
    pub fn decode(&self, encoded: &Array2<f64>) -> Array2<f64> {
        let n_nodes = encoded.dim().0;

        let mut out = Array2::zeros((n_nodes, self.feature_dim));
        for i in 0..n_nodes {
            for d in 0..self.feature_dim {
                let mut val = 0.0;
                for k in 0..self.config.encoder_dim {
                    val += encoded[[i, k]] * self.decoder_weight[[k, d]];
                }
                out[[i, d]] = val;
            }
        }
        out
    }

    /// Scaled Cosine Error (SCE) reconstruction loss on masked nodes.
    ///
    /// ```text
    /// L = (1/|M|) Σ_{i∈M} (1 - cosine_sim(reconstructed_i, original_i))^γ
    /// ```
    ///
    /// If `mask_indices` is empty, returns `0.0`.
    ///
    /// # Arguments
    /// * `original`       – original node features `[n_nodes × feature_dim]`
    /// * `reconstructed`  – decoder output `[n_nodes × feature_dim]`
    /// * `mask_indices`   – indices of masked nodes
    /// * `gamma`          – exponent ≥ 1 (typical: 2 or 3)
    pub fn sce_loss(
        &self,
        original: &Array2<f64>,
        reconstructed: &Array2<f64>,
        mask_indices: &[usize],
        gamma: f64,
    ) -> f64 {
        if mask_indices.is_empty() {
            return 0.0;
        }

        let mut total = 0.0;
        let d = self.feature_dim;

        for &i in mask_indices {
            // Dot product and norms
            let mut dot = 0.0;
            let mut norm_r = 0.0;
            let mut norm_o = 0.0;
            for k in 0..d {
                let r = reconstructed[[i, k]];
                let o = original[[i, k]];
                dot += r * o;
                norm_r += r * r;
                norm_o += o * o;
            }
            let denom = norm_r.sqrt().max(1e-12) * norm_o.sqrt().max(1e-12);
            let cos_sim = (dot / denom).clamp(-1.0, 1.0);
            // SCE: (1 - cos_sim)^gamma
            let term = (1.0 - cos_sim).powf(gamma);
            total += term;
        }

        total / mask_indices.len() as f64
    }

    /// Full GraphMAE forward pass.
    ///
    /// 1. Mask features randomly.
    /// 2. Encode masked features.
    /// 3. Decode back to feature space.
    /// 4. Compute SCE loss over masked nodes (γ = 2).
    ///
    /// # Arguments
    /// * `features` – original node feature matrix `[n_nodes × feature_dim]`
    /// * `seed`     – RNG seed for the masking step
    ///
    /// # Returns
    /// `(reconstructed_features, sce_loss)`
    pub fn forward(&self, features: &Array2<f64>, seed: u64) -> (Array2<f64>, f64) {
        let (masked, mask_indices) = self.mask_features(features, seed);
        let encoded = self.encode(&masked);
        let reconstructed = self.decode(&encoded);
        let loss = self.sce_loss(features, &reconstructed, &mask_indices, 2.0);
        (reconstructed, loss)
    }

    /// The learnable mask token vector `[feature_dim]`.
    pub fn mask_token(&self) -> &Array1<f64> {
        &self.mask_token
    }

    /// Input / output feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Encoder output dimension.
    pub fn encoder_dim(&self) -> usize {
        self.config.encoder_dim
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mae(feature_dim: usize, mask_rate: f64) -> GraphMae {
        let cfg = GraphMaeConfig {
            mask_rate,
            encoder_dim: 16,
            decoder_dim: feature_dim,
            replace_token_scale: 0.1,
        };
        GraphMae::new(feature_dim, cfg, 42)
    }

    #[test]
    fn test_mask_features_approximate_rate() {
        let mae = make_mae(8, 0.5);
        let x = Array2::ones((100, 8));
        let (_, mask_idx) = mae.mask_features(&x, 0);
        // With mask_rate=0.5 and 100 nodes, expected ~50 masked;
        // allow generous tolerance for a stochastic test
        let frac = mask_idx.len() as f64 / 100.0;
        assert!(
            (frac - 0.5).abs() < 0.2,
            "masking fraction {frac} too far from 0.5"
        );
    }

    #[test]
    fn test_encode_output_shape() {
        let mae = make_mae(8, 0.25);
        let x = Array2::ones((10, 8));
        let z = mae.encode(&x);
        assert_eq!(z.dim(), (10, 16));
    }

    #[test]
    fn test_decode_output_shape_matches_feature_dim() {
        let mae = make_mae(8, 0.25);
        let z = Array2::ones((10, 16));
        let out = mae.decode(&z);
        assert_eq!(out.dim(), (10, 8));
    }

    #[test]
    fn test_sce_loss_identical_is_zero() {
        let mae = make_mae(4, 0.25);
        let x = Array2::from_shape_fn((6, 4), |(i, j)| (i + j + 1) as f64);
        // Compute something that equals x by passing x as reconstructed
        let loss = mae.sce_loss(&x, &x, &[0, 1, 2, 3, 4, 5], 2.0);
        assert!(
            loss.abs() < 1e-9,
            "SCE loss for identical tensors should be ~0, got {loss}"
        );
    }

    #[test]
    fn test_sce_loss_orthogonal_positive() {
        let mae = make_mae(4, 0.25);
        // Two orthogonal vectors: cos_sim = 0 → loss per element = 1.0
        let original = Array2::from_shape_vec((2, 4), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            .expect("ok");
        let recon = Array2::from_shape_vec((2, 4), vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            .expect("ok");
        let loss = mae.sce_loss(&original, &recon, &[0, 1], 2.0);
        // cos_sim = 0 → (1-0)^2 = 1 for each node
        assert!(
            (loss - 1.0).abs() < 1e-9,
            "SCE loss for orthogonal vectors should be 1.0, got {loss}"
        );
    }

    #[test]
    fn test_forward_output_shape_consistency() {
        let mae = make_mae(8, 0.25);
        let x = Array2::ones((12, 8));
        let (recon, _loss) = mae.forward(&x, 0);
        assert_eq!(recon.dim(), (12, 8));
    }

    #[test]
    fn test_mask_rate_zero_nothing_masked() {
        let mae = make_mae(4, 0.0);
        let x = Array2::ones((20, 4));
        let (_, idx) = mae.mask_features(&x, 0);
        assert!(idx.is_empty(), "mask_rate=0 should mask no nodes");
        // Loss should be 0 since no nodes are masked
        let encoded = mae.encode(&x);
        let recon = mae.decode(&encoded);
        let loss = mae.sce_loss(&x, &recon, &idx, 2.0);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_mask_rate_one_all_masked() {
        let mae = make_mae(4, 1.0);
        let x = Array2::ones((10, 4));
        let (_, idx) = mae.mask_features(&x, 0);
        assert_eq!(idx.len(), 10, "mask_rate=1 should mask all nodes");
    }

    #[test]
    fn test_forward_loss_is_finite() {
        let mae = make_mae(8, 0.3);
        let x = Array2::from_shape_fn((20, 8), |(i, j)| (i as f64 * 0.1) + (j as f64 * 0.01));
        let (_recon, loss) = mae.forward(&x, 7);
        assert!(loss.is_finite(), "forward loss must be finite, got {loss}");
        assert!(loss >= 0.0, "SCE loss must be non-negative, got {loss}");
    }
}
