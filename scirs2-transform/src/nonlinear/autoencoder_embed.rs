//! Autoencoder-based Dimensionality Reduction
//!
//! This module implements a symmetric stacked autoencoder with tanh activations
//! and mini-batch SGD training.  The encoder maps high-dimensional data to a
//! low-dimensional latent code; the decoder reconstructs the original space.
//!
//! ## Model Architecture
//!
//! Given `encoder_dims = [d0, h1, h2]` and `n_latent = k`:
//!
//! ```text
//! Input (d0) → Dense+tanh (h1) → Dense+tanh (h2) → Dense (k)  [encoder]
//! Latent (k) → Dense+tanh (h2) → Dense+tanh (h1) → Dense (d0) [decoder]
//! ```
//!
//! ## Training
//!
//! Mini-batch SGD minimizing mean squared reconstruction error.
//! Running batch statistics are tracked for monitoring.
//!
//! ## References
//!
//! - Hinton, G.E., & Salakhutdinov, R.R. (2006). Reducing the Dimensionality
//!   of Data with Neural Networks. Science.
//! - Vincent, P., et al. (2008). Extracting and Composing Robust Features with
//!   Denoising Autoencoders. ICML.

use crate::error::{Result, TransformError};
use scirs2_core::random::{seeded_rng, Distribution, Normal, RngCore, SeedableRng};

// ============================================================================
// Layer representation
// ============================================================================

/// A fully-connected layer: y = W x + b.
#[derive(Debug, Clone)]
pub struct AELayer {
    /// Weight matrix: shape (out_dim × in_dim).
    pub weights: Vec<Vec<f64>>,
    /// Bias vector: shape (out_dim).
    pub biases: Vec<f64>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

impl AELayer {
    /// Create a new layer with Xavier (Glorot) uniform initialization.
    fn new_xavier(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        let mut rng = seeded_rng(seed);
        let scale = (6.0_f64 / (in_dim + out_dim) as f64).sqrt();
        let dist = Normal::new(0.0, scale / 3.0_f64.sqrt())
            .expect("Normal::new failed in AELayer::new_xavier");

        let weights: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| (0..in_dim).map(|_| dist.sample(&mut rng)).collect())
            .collect();
        let biases = vec![0.0f64; out_dim];

        AELayer { weights, biases, in_dim, out_dim }
    }

    /// Forward pass: computes pre-activation (before nonlinearity).
    fn forward_pre(&self, x: &[f64]) -> Vec<f64> {
        let mut out = self.biases.clone();
        for (i, row) in self.weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                out[i] += w * x[j];
            }
        }
        out
    }
}

// ============================================================================
// Activation functions
// ============================================================================

#[inline]
fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[inline]
fn tanh_deriv(tanh_val: f64) -> f64 {
    1.0 - tanh_val * tanh_val
}

// ============================================================================
// Autoencoder embedder
// ============================================================================

/// Autoencoder-based dimensionality reduction.
///
/// Architecture:
/// - Encoder: `encoder_dims[0]` → hidden layers → `n_latent`
/// - Decoder: `n_latent` → hidden layers (reversed) → `encoder_dims[0]`
///
/// Training uses mini-batch SGD with MSE loss.
#[derive(Debug, Clone)]
pub struct AEEmbedder {
    /// Encoder hidden dimensions (not including input dim or latent).
    pub encoder_dims: Vec<usize>,
    /// Input / output dimension.
    pub input_dim: usize,
    /// Latent dimension.
    pub n_latent: usize,
    /// Encoder layers (including final projection to latent).
    encoder_layers: Vec<AELayer>,
    /// Decoder layers (including final reconstruction layer).
    decoder_layers: Vec<AELayer>,
    /// Whether the model has been trained.
    fitted: bool,
    /// Running mean of reconstruction error during training.
    pub train_loss: f64,
}

impl AEEmbedder {
    /// Create a new (unfitted) autoencoder embedder.
    ///
    /// `encoder_dims`: sequence of hidden layer widths for the encoder, e.g. `[256, 128]`.
    /// The decoder will mirror these in reverse.
    pub fn new(input_dim: usize, encoder_dims: Vec<usize>, n_latent: usize) -> Result<Self> {
        if input_dim == 0 {
            return Err(TransformError::InvalidInput(
                "input_dim must be > 0".to_string(),
            ));
        }
        if n_latent == 0 {
            return Err(TransformError::InvalidInput(
                "n_latent must be > 0".to_string(),
            ));
        }

        Ok(AEEmbedder {
            encoder_dims,
            input_dim,
            n_latent,
            encoder_layers: vec![],
            decoder_layers: vec![],
            fitted: false,
            train_loss: f64::INFINITY,
        })
    }

    /// Build encoder and decoder layer stacks.
    fn build_layers(&mut self, seed: u64) {
        // Encoder: input_dim → h1 → h2 → ... → n_latent
        let mut enc_dims: Vec<usize> = vec![self.input_dim];
        enc_dims.extend_from_slice(&self.encoder_dims);
        enc_dims.push(self.n_latent);

        self.encoder_layers = enc_dims
            .windows(2)
            .enumerate()
            .map(|(i, w)| AELayer::new_xavier(w[0], w[1], seed + i as u64))
            .collect();

        // Decoder: n_latent → h2 → h1 → input_dim  (reversed encoder hidden dims)
        let mut dec_dims: Vec<usize> = vec![self.n_latent];
        let rev_hidden: Vec<usize> = self.encoder_dims.iter().cloned().rev().collect();
        dec_dims.extend_from_slice(&rev_hidden);
        dec_dims.push(self.input_dim);

        let offset = self.encoder_layers.len();
        self.decoder_layers = dec_dims
            .windows(2)
            .enumerate()
            .map(|(i, w)| AELayer::new_xavier(w[0], w[1], seed + offset as u64 + i as u64))
            .collect();
    }

    /// Encode a single sample through all encoder layers.
    /// Returns latent code.
    fn encode_one(&self, x: &[f64]) -> Vec<f64> {
        let n_enc = self.encoder_layers.len();
        let mut h = x.to_vec();
        for (li, layer) in self.encoder_layers.iter().enumerate() {
            let pre = layer.forward_pre(&h);
            // Apply tanh to all layers except the last (latent) layer
            if li < n_enc - 1 {
                h = pre.iter().map(|&v| tanh(v)).collect();
            } else {
                h = pre; // linear activation for latent
            }
        }
        h
    }

    /// Decode a single latent code through all decoder layers.
    fn decode_one(&self, z: &[f64]) -> Vec<f64> {
        let n_dec = self.decoder_layers.len();
        let mut h = z.to_vec();
        for (li, layer) in self.decoder_layers.iter().enumerate() {
            let pre = layer.forward_pre(&h);
            // Apply tanh to all layers except the last (reconstruction) layer
            if li < n_dec - 1 {
                h = pre.iter().map(|&v| tanh(v)).collect();
            } else {
                h = pre; // linear output
            }
        }
        h
    }

    /// Forward pass returning (latent, reconstructed, per-layer pre-activations and activations).
    ///
    /// Returns:
    /// - `enc_pres`: pre-activation values for each encoder layer
    /// - `enc_acts`: activation values after each encoder layer (including latent)
    /// - `dec_pres`: pre-activation values for each decoder layer
    /// - `dec_acts`: activation values after each decoder layer
    fn forward_cache(
        &self,
        x: &[f64],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n_enc = self.encoder_layers.len();
        let n_dec = self.decoder_layers.len();

        let mut enc_pres = Vec::with_capacity(n_enc);
        let mut enc_acts: Vec<Vec<f64>> = Vec::with_capacity(n_enc + 1);
        enc_acts.push(x.to_vec()); // input as "activation before layer 0"

        let mut h = x.to_vec();
        for (li, layer) in self.encoder_layers.iter().enumerate() {
            let pre = layer.forward_pre(&h);
            let act: Vec<f64> = if li < n_enc - 1 {
                pre.iter().map(|&v| tanh(v)).collect()
            } else {
                pre.clone()
            };
            enc_pres.push(pre);
            enc_acts.push(act.clone());
            h = act;
        }

        let mut dec_pres = Vec::with_capacity(n_dec);
        let mut dec_acts: Vec<Vec<f64>> = Vec::with_capacity(n_dec + 1);
        dec_acts.push(h.clone()); // latent as "activation before decoder layer 0"

        for (li, layer) in self.decoder_layers.iter().enumerate() {
            let pre = layer.forward_pre(&h);
            let act: Vec<f64> = if li < n_dec - 1 {
                pre.iter().map(|&v| tanh(v)).collect()
            } else {
                pre.clone()
            };
            dec_pres.push(pre);
            dec_acts.push(act.clone());
            h = act;
        }

        (enc_pres, enc_acts, dec_pres, dec_acts)
    }

    /// Backpropagation: compute gradients for all layers given input `x`.
    /// Returns (enc_weight_grads, enc_bias_grads, dec_weight_grads, dec_bias_grads).
    fn backprop(
        &self,
        x: &[f64],
    ) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let n_enc = self.encoder_layers.len();
        let n_dec = self.decoder_layers.len();

        let (enc_pres, enc_acts, dec_pres, dec_acts) = self.forward_cache(x);

        // Reconstruction = dec_acts[n_dec]
        let recon = &dec_acts[n_dec];
        let input_dim = x.len();

        // MSE loss gradient w.r.t. reconstruction: dL/d_recon = 2/d * (recon - x)
        let scale = 2.0 / input_dim as f64;
        let mut delta: Vec<f64> = recon
            .iter()
            .zip(x.iter())
            .map(|(r, xi)| scale * (r - xi))
            .collect();

        // Decoder gradients (backprop through decoder layers in reverse)
        let mut dec_wgrads: Vec<Vec<Vec<f64>>> = vec![vec![]; n_dec];
        let mut dec_bgrads: Vec<Vec<f64>> = vec![vec![]; n_dec];

        for li in (0..n_dec).rev() {
            let layer = &self.decoder_layers[li];
            // Activation derivative
            let act_delta: Vec<f64> = if li < n_dec - 1 {
                // tanh layers
                let tanh_vals: Vec<f64> = dec_pres[li].iter().map(|&v| tanh(v)).collect();
                delta
                    .iter()
                    .zip(tanh_vals.iter())
                    .map(|(d, tv)| d * tanh_deriv(*tv))
                    .collect()
            } else {
                // linear output layer
                delta.clone()
            };

            let input_act = &dec_acts[li]; // activation feeding into this layer

            // Weight gradient: delta_a ⊗ input
            let wg: Vec<Vec<f64>> = act_delta
                .iter()
                .map(|&da| input_act.iter().map(|&ia| da * ia).collect())
                .collect();
            let bg = act_delta.clone();

            // Propagate delta to previous layer
            let prev_delta: Vec<f64> = (0..layer.in_dim)
                .map(|j| {
                    layer
                        .weights
                        .iter()
                        .zip(act_delta.iter())
                        .map(|(row, &da)| row[j] * da)
                        .sum::<f64>()
                })
                .collect();

            dec_wgrads[li] = wg;
            dec_bgrads[li] = bg;
            delta = prev_delta;
        }

        // Encoder gradients (backprop through encoder layers in reverse)
        let mut enc_wgrads: Vec<Vec<Vec<f64>>> = vec![vec![]; n_enc];
        let mut enc_bgrads: Vec<Vec<f64>> = vec![vec![]; n_enc];

        for li in (0..n_enc).rev() {
            let layer = &self.encoder_layers[li];
            let act_delta: Vec<f64> = if li < n_enc - 1 {
                let tanh_vals: Vec<f64> = enc_pres[li].iter().map(|&v| tanh(v)).collect();
                delta
                    .iter()
                    .zip(tanh_vals.iter())
                    .map(|(d, tv)| d * tanh_deriv(*tv))
                    .collect()
            } else {
                // latent layer is linear
                delta.clone()
            };

            let input_act = &enc_acts[li]; // activation feeding into encoder layer li

            let wg: Vec<Vec<f64>> = act_delta
                .iter()
                .map(|&da| input_act.iter().map(|&ia| da * ia).collect())
                .collect();
            let bg = act_delta.clone();

            let prev_delta: Vec<f64> = (0..layer.in_dim)
                .map(|j| {
                    layer
                        .weights
                        .iter()
                        .zip(act_delta.iter())
                        .map(|(row, &da)| row[j] * da)
                        .sum::<f64>()
                })
                .collect();

            enc_wgrads[li] = wg;
            enc_bgrads[li] = bg;
            delta = prev_delta;
        }

        (enc_wgrads, enc_bgrads, dec_wgrads, dec_bgrads)
    }

    /// Apply gradient update (SGD) to all layers.
    fn apply_gradients(
        &mut self,
        enc_wg: &[Vec<Vec<f64>>],
        enc_bg: &[Vec<f64>],
        dec_wg: &[Vec<Vec<f64>>],
        dec_bg: &[Vec<f64>],
        lr: f64,
        batch_size: usize,
    ) {
        let scale = lr / batch_size as f64;

        for (li, layer) in self.encoder_layers.iter_mut().enumerate() {
            if li >= enc_wg.len() {
                break;
            }
            for (oi, row) in layer.weights.iter_mut().enumerate() {
                if oi >= enc_wg[li].len() {
                    break;
                }
                for (ii, w) in row.iter_mut().enumerate() {
                    if ii < enc_wg[li][oi].len() {
                        *w -= scale * enc_wg[li][oi][ii];
                    }
                }
            }
            for (oi, b) in layer.biases.iter_mut().enumerate() {
                if oi < enc_bg[li].len() {
                    *b -= scale * enc_bg[li][oi];
                }
            }
        }

        for (li, layer) in self.decoder_layers.iter_mut().enumerate() {
            if li >= dec_wg.len() {
                break;
            }
            for (oi, row) in layer.weights.iter_mut().enumerate() {
                if oi >= dec_wg[li].len() {
                    break;
                }
                for (ii, w) in row.iter_mut().enumerate() {
                    if ii < dec_wg[li][oi].len() {
                        *w -= scale * dec_wg[li][oi][ii];
                    }
                }
            }
            for (oi, b) in layer.biases.iter_mut().enumerate() {
                if oi < dec_bg[li].len() {
                    *b -= scale * dec_bg[li][oi];
                }
            }
        }
    }

    /// Train the autoencoder via mini-batch SGD.
    ///
    /// # Arguments
    ///
    /// * `x` - Training data (n × d).
    /// * `n_epochs` - Number of full passes over the data.
    /// * `lr` - Learning rate.
    /// * `batch_size` - Mini-batch size.
    /// * `seed` - RNG seed for weight initialization and batch shuffling.
    pub fn fit(
        &mut self,
        x: &[Vec<f64>],
        n_epochs: usize,
        lr: f64,
        batch_size: usize,
        seed: u64,
    ) -> Result<()> {
        let n = x.len();
        if n == 0 {
            return Err(TransformError::InvalidInput("Empty training set".to_string()));
        }

        // Validate dimensions
        for (i, row) in x.iter().enumerate() {
            if row.len() != self.input_dim {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i}: expected {} features, got {}",
                    self.input_dim,
                    row.len()
                )));
            }
        }

        if n_epochs == 0 {
            return Err(TransformError::InvalidInput(
                "n_epochs must be > 0".to_string(),
            ));
        }
        let batch_size = batch_size.max(1).min(n);

        // Build layers (random init)
        self.build_layers(seed);

        let mut rng = seeded_rng(seed + 1000);
        let mut indices: Vec<usize> = (0..n).collect();

        let mut total_loss = 0.0f64;
        let mut total_batches = 0usize;

        for _epoch in 0..n_epochs {
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                indices.swap(i, j);
            }

            let mut start = 0;
            while start < n {
                let end = (start + batch_size).min(n);
                let actual_batch = end - start;

                // Accumulate gradients over batch
                let n_enc = self.encoder_layers.len();
                let n_dec = self.decoder_layers.len();

                // Initialize accumulated gradient buffers
                let mut acc_enc_wg: Vec<Vec<Vec<f64>>> = self
                    .encoder_layers
                    .iter()
                    .map(|l| vec![vec![0.0; l.in_dim]; l.out_dim])
                    .collect();
                let mut acc_enc_bg: Vec<Vec<f64>> = self
                    .encoder_layers
                    .iter()
                    .map(|l| vec![0.0; l.out_dim])
                    .collect();
                let mut acc_dec_wg: Vec<Vec<Vec<f64>>> = self
                    .decoder_layers
                    .iter()
                    .map(|l| vec![vec![0.0; l.in_dim]; l.out_dim])
                    .collect();
                let mut acc_dec_bg: Vec<Vec<f64>> = self
                    .decoder_layers
                    .iter()
                    .map(|l| vec![0.0; l.out_dim])
                    .collect();

                let mut batch_loss = 0.0f64;

                for &idx in &indices[start..end] {
                    let sample = &x[idx];
                    let (ewg, ebg, dwg, dbg) = self.backprop(sample);

                    // Compute reconstruction loss for monitoring
                    let recon = self.decode_one(&self.encode_one(sample));
                    let mse: f64 = sample
                        .iter()
                        .zip(recon.iter())
                        .map(|(xi, ri)| (xi - ri).powi(2))
                        .sum::<f64>()
                        / sample.len() as f64;
                    batch_loss += mse;

                    // Accumulate
                    for li in 0..n_enc.min(ewg.len()) {
                        for oi in 0..acc_enc_wg[li].len().min(ewg[li].len()) {
                            for ii in 0..acc_enc_wg[li][oi].len().min(ewg[li][oi].len()) {
                                acc_enc_wg[li][oi][ii] += ewg[li][oi][ii];
                            }
                        }
                        for oi in 0..acc_enc_bg[li].len().min(ebg[li].len()) {
                            acc_enc_bg[li][oi] += ebg[li][oi];
                        }
                    }
                    for li in 0..n_dec.min(dwg.len()) {
                        for oi in 0..acc_dec_wg[li].len().min(dwg[li].len()) {
                            for ii in 0..acc_dec_wg[li][oi].len().min(dwg[li][oi].len()) {
                                acc_dec_wg[li][oi][ii] += dwg[li][oi][ii];
                            }
                        }
                        for oi in 0..acc_dec_bg[li].len().min(dbg[li].len()) {
                            acc_dec_bg[li][oi] += dbg[li][oi];
                        }
                    }
                }

                self.apply_gradients(&acc_enc_wg, &acc_enc_bg, &acc_dec_wg, &acc_dec_bg, lr, actual_batch);
                total_loss += batch_loss / actual_batch as f64;
                total_batches += 1;
                start = end;
            }
        }

        self.train_loss = if total_batches > 0 { total_loss / total_batches as f64 } else { 0.0 };
        self.fitted = true;
        Ok(())
    }

    /// Encode a batch of samples to latent codes.
    pub fn encode(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(TransformError::NotFitted(
                "AEEmbedder must be fitted before encoding".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(x.len());
        for (i, row) in x.iter().enumerate() {
            if row.len() != self.input_dim {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i}: expected {} features, got {}",
                    self.input_dim,
                    row.len()
                )));
            }
            out.push(self.encode_one(row));
        }
        Ok(out)
    }

    /// Decode latent codes back to the original space.
    pub fn decode(&self, z: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(TransformError::NotFitted(
                "AEEmbedder must be fitted before decoding".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(z.len());
        for (i, row) in z.iter().enumerate() {
            if row.len() != self.n_latent {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i}: expected {} latent dims, got {}",
                    self.n_latent,
                    row.len()
                )));
            }
            out.push(self.decode_one(row));
        }
        Ok(out)
    }

    /// Reconstruct samples and return (reconstructed_x, mse).
    pub fn reconstruct(&self, x: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, f64)> {
        let z = self.encode(x)?;
        let recon = self.decode(&z)?;
        let mse: f64 = x
            .iter()
            .zip(recon.iter())
            .map(|(xi, ri)| {
                xi.iter()
                    .zip(ri.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / xi.len() as f64
            })
            .sum::<f64>()
            / x.len() as f64;
        Ok((recon, mse))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = seeded_rng(seed);
        let dist = Normal::new(0.0_f64, 1.0).expect("Normal");
        (0..n)
            .map(|_| (0..d).map(|_| dist.sample(&mut rng)).collect())
            .collect()
    }

    #[test]
    fn test_ae_fit_encode_decode() {
        let x = make_data(50, 8, 42);
        let mut ae = AEEmbedder::new(8, vec![4], 2).expect("AEEmbedder::new");
        ae.fit(&x, 5, 0.01, 16, 0).expect("fit");

        let z = ae.encode(&x).expect("encode");
        assert_eq!(z.len(), 50);
        assert_eq!(z[0].len(), 2);

        let xhat = ae.decode(&z).expect("decode");
        assert_eq!(xhat.len(), 50);
        assert_eq!(xhat[0].len(), 8);
    }

    #[test]
    fn test_ae_reconstruct() {
        let x = make_data(30, 6, 7);
        let mut ae = AEEmbedder::new(6, vec![3], 2).expect("new");
        ae.fit(&x, 5, 0.01, 8, 1).expect("fit");

        let (recon, mse) = ae.reconstruct(&x).expect("reconstruct");
        assert_eq!(recon.len(), 30);
        assert!(mse >= 0.0, "MSE should be non-negative");
    }

    #[test]
    fn test_ae_not_fitted_error() {
        let x = make_data(10, 4, 0);
        let ae = AEEmbedder::new(4, vec![2], 1).expect("new");
        assert!(ae.encode(&x).is_err());
        assert!(ae.decode(&[vec![0.0]]).is_err());
    }

    #[test]
    fn test_ae_deep_architecture() {
        let x = make_data(40, 16, 99);
        let mut ae = AEEmbedder::new(16, vec![12, 8], 4).expect("new");
        ae.fit(&x, 3, 0.005, 10, 5).expect("fit");

        let z = ae.encode(&x).expect("encode");
        assert_eq!(z[0].len(), 4);

        let (_recon, mse) = ae.reconstruct(&x).expect("reconstruct");
        assert!(mse.is_finite());
    }

    #[test]
    fn test_ae_dimension_mismatch() {
        let x = make_data(20, 4, 0);
        let mut ae = AEEmbedder::new(4, vec![2], 2).expect("new");
        ae.fit(&x, 3, 0.01, 5, 0).expect("fit");

        let bad_input = vec![vec![1.0, 2.0, 3.0]]; // wrong dim
        assert!(ae.encode(&bad_input).is_err());
    }
}
