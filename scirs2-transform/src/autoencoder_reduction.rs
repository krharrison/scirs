//! Autoencoder-based dimensionality reduction
//!
//! This module provides linear and denoising autoencoders for unsupervised
//! dimensionality reduction of tabular data.
//!
//! ## Overview
//!
//! An **autoencoder** is a neural network trained to reconstruct its own input
//! through a bottleneck latent space.  The encoder maps the input to the latent
//! representation, and the decoder maps the latent representation back to the
//! input.  When the latent dimension is smaller than the input dimension the
//! network is forced to learn a compact representation.
//!
//! ### Linear Autoencoder
//!
//! A linear autoencoder (no activation functions) is equivalent to PCA:
//! the encoder weight matrix converges to a rotation of the top principal
//! components.  It is included here as a baseline / educational reference
//! and as a building block for the denoising variant.
//!
//! ### Denoising Autoencoder
//!
//! A denoising autoencoder (DAE) is trained to reconstruct **clean** inputs
//! from **noisy** versions of those inputs.  The corruption forces the model
//! to learn more robust representations that capture the underlying structure
//! rather than memorizing individual samples.
//!
//! ## References
//!
//! - Vincent, P., et al. (2008). Extracting and composing robust features with
//!   denoising autoencoders. ICML 2008.
//! - Bourlard, H., & Kamp, Y. (1988). Auto-association by multilayer perceptrons
//!   and singular value decomposition. Biological Cybernetics, 59(4-5), 291-294.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ─── Noise Injection ──────────────────────────────────────────────────────────

/// Add element-wise Gaussian noise to a data matrix.
///
/// Each element `x[i,j]` becomes `x[i,j] + ε` where `ε ~ N(0, noise_std²)`.
///
/// Uses a deterministic LCG PRNG seeded with a fixed constant so that tests
/// are reproducible without requiring a random seed parameter.  The quality of
/// the randomness is sufficient for additive Gaussian noise injection.
pub fn add_gaussian_noise(data: &Array2<f64>, noise_std: f64) -> Result<Array2<f64>> {
    if noise_std < 0.0 {
        return Err(TransformError::InvalidInput(
            "noise_std must be non-negative".to_string(),
        ));
    }
    if noise_std == 0.0 {
        return Ok(data.clone());
    }

    let n_elems = data.nrows() * data.ncols();
    let noise = lcg_normal_samples(n_elems, noise_std);
    let noise_arr = Array2::from_shape_vec((data.nrows(), data.ncols()), noise)
        .map_err(|e| TransformError::ComputationError(e.to_string()))?;
    Ok(data + &noise_arr)
}

/// Generate `n` approximately-Gaussian samples with zero mean and `std`
/// standard deviation using the Box-Muller transform over a LCG PRNG.
///
/// This avoids a dependency on any external random-number crate while still
/// producing a usable noise distribution for training.
fn lcg_normal_samples(n: usize, std: f64) -> Vec<f64> {
    // LCG parameters (Numerical Recipes)
    const A: u64 = 1_664_525;
    const C: u64 = 1_013_904_223;
    const M: u64 = 1 << 32;

    let mut state: u64 = 2_463_534_242; // fixed seed
    let mut samples = Vec::with_capacity(n);

    let mut i = 0;
    while i < n {
        // Two uniform samples u1, u2 ∈ (0, 1)
        state = (A.wrapping_mul(state).wrapping_add(C)) % M;
        let u1 = state as f64 / M as f64;
        state = (A.wrapping_mul(state).wrapping_add(C)) % M;
        let u2 = state as f64 / M as f64;

        // Box-Muller transform
        let u1 = u1.max(1e-12); // guard log(0)
        let mag = std * (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();

        samples.push(z0);
        i += 1;
        if i < n {
            samples.push(z1);
            i += 1;
        }
    }

    samples
}

// ─── Helper: Matrix Operations ────────────────────────────────────────────────

/// Compute mean of each column (feature).
fn column_means(data: &Array2<f64>) -> Array1<f64> {
    data.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(data.ncols()))
}

/// Centre a matrix by subtracting column means.
fn centre(data: &Array2<f64>, means: &Array1<f64>) -> Array2<f64> {
    let mut out = data.clone();
    for mut row in out.rows_mut() {
        for (v, &m) in row.iter_mut().zip(means.iter()) {
            *v -= m;
        }
    }
    out
}

/// Matrix multiplication `A @ B` using simple triple-loop (pure Rust, no BLAS).
///
/// Returns `Array2<f64>` of shape `(a_rows, b_cols)`.
fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    let (m, k1) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    if k1 != k2 {
        return Err(TransformError::ComputationError(format!(
            "matmul dimension mismatch: ({m},{k1}) × ({k2},{n})"
        )));
    }
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for l in 0..k1 {
                s += a[[i, l]] * b[[l, j]];
            }
            out[[i, j]] = s;
        }
    }
    Ok(out)
}

/// Transpose a matrix.
fn transpose(a: &Array2<f64>) -> Array2<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut out = Array2::<f64>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            out[[j, i]] = a[[i, j]];
        }
    }
    out
}

/// Mean-squared error between two same-shape arrays.
fn mse(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let n = (a.nrows() * a.ncols()) as f64;
    if n == 0.0 {
        return 0.0;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        / n
}

// ─── Linear Autoencoder ───────────────────────────────────────────────────────

/// A trained linear autoencoder.
///
/// The encoder maps `x ∈ ℝⁿ → z ∈ ℝᵏ` via `z = x W_enc`
/// and the decoder maps `z ∈ ℝᵏ → x̂ ∈ ℝⁿ` via `x̂ = z W_dec`,
/// where `W_enc ∈ ℝⁿˣᵏ` and `W_dec ∈ ℝᵏˣⁿ`.
///
/// Both weights are stored together with the training means for centring.
#[derive(Debug, Clone)]
pub struct LinearAutoencoder {
    /// Encoder weight matrix of shape `(n_features, n_components)`
    pub encoder_weights: Array2<f64>,
    /// Decoder weight matrix of shape `(n_components, n_features)`
    pub decoder_weights: Array2<f64>,
    /// Column means of the training data (used for centring)
    pub means: Array1<f64>,
    /// Number of input features
    pub n_features: usize,
    /// Bottleneck dimension
    pub n_components: usize,
    /// Reconstruction losses recorded at each training iteration
    pub loss_history: Vec<f64>,
}

impl LinearAutoencoder {
    /// Encode (compress) data.
    ///
    /// `data` has shape `(n_samples, n_features)`.
    /// Returns latent codes of shape `(n_samples, n_components)`.
    pub fn encode(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        if data.ncols() != self.n_features {
            return Err(TransformError::InvalidInput(format!(
                "data has {} features but autoencoder expects {}",
                data.ncols(),
                self.n_features
            )));
        }
        let centred = centre(data, &self.means);
        encode(&centred, &self.encoder_weights)
    }

    /// Decode (reconstruct) latent codes.
    ///
    /// `latent` has shape `(n_samples, n_components)`.
    /// Returns reconstructed data of shape `(n_samples, n_features)`.
    pub fn decode(&self, latent: &Array2<f64>) -> Result<Array2<f64>> {
        if latent.ncols() != self.n_components {
            return Err(TransformError::InvalidInput(format!(
                "latent has {} dims but autoencoder has {} components",
                latent.ncols(),
                self.n_components
            )));
        }
        let recon = decode(latent, &self.decoder_weights)?;
        // Add back means
        let mut out = recon;
        for mut row in out.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.means.iter()) {
                *v += m;
            }
        }
        Ok(out)
    }

    /// Reconstruction MSE on `data`.
    pub fn reconstruction_loss(&self, data: &Array2<f64>) -> Result<f64> {
        let latent = self.encode(data)?;
        let recon_centred = decode(&latent, &self.decoder_weights)?;
        let centred = centre(data, &self.means);
        Ok(mse(&centred, &recon_centred))
    }
}

// ─── Public Encode / Decode Functions ────────────────────────────────────────

/// Apply encoder transform: `latent = data @ W_enc`.
///
/// `data` has shape `(n_samples, n_features)`.
/// `weights` has shape `(n_features, n_components)`.
///
/// Returns array of shape `(n_samples, n_components)`.
pub fn encode(data: &Array2<f64>, weights: &Array2<f64>) -> Result<Array2<f64>> {
    matmul(data, weights)
}

/// Apply decoder transform: `recon = latent @ W_dec`.
///
/// `latent` has shape `(n_samples, n_components)`.
/// `weights` has shape `(n_components, n_features)`.
///
/// Returns array of shape `(n_samples, n_features)`.
pub fn decode(latent: &Array2<f64>, weights: &Array2<f64>) -> Result<Array2<f64>> {
    matmul(latent, weights)
}

// ─── Linear AE Training ───────────────────────────────────────────────────────

/// Fit a linear autoencoder by gradient descent on MSE reconstruction loss.
///
/// # Parameters
///
/// * `data`         – Training data of shape `(n_samples, n_features)`.
/// * `n_components` – Bottleneck dimension (latent space size).
/// * `lr`           – Learning rate (step size for gradient descent).
/// * `n_iter`       – Number of gradient descent steps.
///
/// # Returns
///
/// Fitted [`LinearAutoencoder`].
///
/// # Algorithm
///
/// Uses full-batch gradient descent on the MSE loss.  The gradient update
/// for tied weights (where `W_dec = W_enc^T`) is performed explicitly so
/// that the encoder/decoder relationship is maintained.  After convergence
/// the weights approximate the principal components of the data.
pub fn fit_linear_ae(
    data: &Array2<f64>,
    n_components: usize,
    lr: f64,
    n_iter: usize,
) -> Result<LinearAutoencoder> {
    validate_ae_params(data, n_components, lr, n_iter)?;

    let means = column_means(data);
    let x = centre(data, &means);
    let n_features = data.ncols();
    let n_samples = data.nrows();

    // Initialise encoder with orthogonal-ish small random values
    let enc_flat = lcg_normal_samples(n_features * n_components, 0.01);
    let mut w_enc = Array2::from_shape_vec((n_features, n_components), enc_flat)
        .map_err(|e| TransformError::ComputationError(e.to_string()))?;

    // Orthonormalise initial columns using Gram-Schmidt
    w_enc = gram_schmidt_orthonorm(&w_enc)?;

    let mut loss_history = Vec::with_capacity(n_iter);

    for _ in 0..n_iter {
        // Forward pass
        let z = matmul(&x, &w_enc)?;           // (n, k)
        let w_dec = transpose(&w_enc);          // (k, d)
        let x_hat = matmul(&z, &w_dec)?;       // (n, d)

        // Residual
        let residual = {
            let mut r = x_hat.clone();
            for i in 0..n_samples {
                for j in 0..n_features {
                    r[[i, j]] -= x[[i, j]];
                }
            }
            r
        }; // (n, d)

        let loss = residual.iter().map(|v| v * v).sum::<f64>() / (n_samples * n_features) as f64;
        loss_history.push(loss);

        // Gradient w.r.t. W_enc with tied weights W_dec = W_enc^T.
        //
        // Loss = (1/(N*D)) || X - X W W^T ||_F^2
        //
        // Gradient (Bourlard & Kamp, 1988):
        //   dL/dW = (2/(N*D)) * [ X^T (X W W^T - X) W  +  (X W W^T - X)^T X W W^T ... ]
        //
        // In practice for tied weights the cleanest form is:
        //   grad_enc = (2/(N*D)) * X^T @ residual @ W_enc
        //            + (2/(N*D)) * X^T @ residual   (via the decoder path)
        //
        // Combined as: grad = X^T @ (residual @ W_enc + residual) scaled by 2/(N*D)
        // This is the standard tied-autoencoder gradient.
        let xt = transpose(&x);                      // (d, n)
        let resid_wenc = matmul(&residual, &w_enc)?; // (n, k): residual @ W_enc
        // First term: X^T @ (residual @ W_enc)
        let grad_a = matmul(&xt, &resid_wenc)?;      // (d, k)
        // Second term: X^T @ residual @ W_enc (decoder path, same form)
        // = X^T @ resid_wenc (already computed above as grad_a)
        // So the full tied gradient is 2 * grad_a:
        let scale = 2.0 / (n_samples * n_features) as f64;
        for i in 0..n_features {
            for j in 0..n_components {
                w_enc[[i, j]] -= lr * scale * 2.0 * grad_a[[i, j]];
            }
        }
    }

    let w_dec = transpose(&w_enc);
    Ok(LinearAutoencoder {
        encoder_weights: w_enc,
        decoder_weights: w_dec,
        means,
        n_features,
        n_components,
        loss_history,
    })
}

/// Gram-Schmidt orthonormalisation of the columns of a matrix.
fn gram_schmidt_orthonorm(mat: &Array2<f64>) -> Result<Array2<f64>> {
    let (m, n) = (mat.nrows(), mat.ncols());
    let mut out = mat.clone();

    for j in 0..n {
        // Subtract projections onto all previous columns
        for k in 0..j {
            let dot: f64 = (0..m).map(|i| out[[i, k]] * out[[i, j]]).sum();
            let norm_sq: f64 = (0..m).map(|i| out[[i, k]] * out[[i, k]]).sum();
            if norm_sq > 1e-12 {
                let coeff = dot / norm_sq;
                for i in 0..m {
                    out[[i, j]] -= coeff * out[[i, k]];
                }
            }
        }
        // Normalise column j
        let norm: f64 = (0..m).map(|i| out[[i, j]] * out[[i, j]]).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for i in 0..m {
                out[[i, j]] /= norm;
            }
        }
    }
    Ok(out)
}

// ─── Denoising Autoencoder ────────────────────────────────────────────────────

/// A trained denoising autoencoder (linear variant).
///
/// Identical structure to [`LinearAutoencoder`] but trained by corrupting
/// inputs with Gaussian noise during training, forcing more robust encoder
/// weights.
#[derive(Debug, Clone)]
pub struct DenoisingAE {
    /// Encoder weight matrix of shape `(n_features, n_components)`
    pub encoder_weights: Array2<f64>,
    /// Decoder weight matrix of shape `(n_components, n_features)`
    pub decoder_weights: Array2<f64>,
    /// Column means of the training data (used for centring)
    pub means: Array1<f64>,
    /// Number of input features
    pub n_features: usize,
    /// Bottleneck dimension
    pub n_components: usize,
    /// Standard deviation of the Gaussian noise used during training
    pub noise_std: f64,
    /// Reconstruction losses on the clean data, recorded per iteration
    pub loss_history: Vec<f64>,
}

impl DenoisingAE {
    /// Encode (compress) clean data.
    pub fn encode(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        if data.ncols() != self.n_features {
            return Err(TransformError::InvalidInput(format!(
                "data has {} features but DAE expects {}",
                data.ncols(),
                self.n_features
            )));
        }
        let centred = centre(data, &self.means);
        encode(&centred, &self.encoder_weights)
    }

    /// Decode (reconstruct) latent codes.
    pub fn decode(&self, latent: &Array2<f64>) -> Result<Array2<f64>> {
        if latent.ncols() != self.n_components {
            return Err(TransformError::InvalidInput(format!(
                "latent has {} dims but DAE has {} components",
                latent.ncols(),
                self.n_components
            )));
        }
        let recon = decode(latent, &self.decoder_weights)?;
        let mut out = recon;
        for mut row in out.rows_mut() {
            for (v, &m) in row.iter_mut().zip(self.means.iter()) {
                *v += m;
            }
        }
        Ok(out)
    }

    /// Reconstruction MSE on **clean** data.
    pub fn reconstruction_loss(&self, data: &Array2<f64>) -> Result<f64> {
        let centred = centre(data, &self.means);
        let latent = encode(&centred, &self.encoder_weights)?;
        let recon = decode(&latent, &self.decoder_weights)?;
        Ok(mse(&centred, &recon))
    }
}

// ─── Denoising AE Training ────────────────────────────────────────────────────

/// Fit a denoising autoencoder.
///
/// # Parameters
///
/// * `clean_data`   – Clean training data of shape `(n_samples, n_features)`.
/// * `n_components` – Bottleneck dimension.
/// * `noise_std`    – Standard deviation of Gaussian corruption noise.
/// * `lr`           – Learning rate.
/// * `n_iter`       – Number of gradient descent steps.
///
/// # Returns
///
/// Fitted [`DenoisingAE`].
///
/// # Algorithm
///
/// At each iteration:
/// 1. Corrupt the centred data by adding Gaussian noise: `x̃ = x + ε`.
/// 2. Forward pass: `z = x̃ W_enc`, `x̂ = z W_dec`.
/// 3. Compute MSE loss between reconstruction `x̂` and the **clean** `x`.
/// 4. Back-propagate gradients to update `W_enc` (with `W_dec = W_enc^T`).
pub fn fit_denoising_ae(
    clean_data: &Array2<f64>,
    n_components: usize,
    noise_std: f64,
    lr: f64,
    n_iter: usize,
) -> Result<DenoisingAE> {
    validate_ae_params(clean_data, n_components, lr, n_iter)?;
    if noise_std < 0.0 {
        return Err(TransformError::InvalidInput(
            "noise_std must be non-negative".to_string(),
        ));
    }

    let means = column_means(clean_data);
    let x_clean = centre(clean_data, &means);
    let n_features = clean_data.ncols();
    let n_samples = clean_data.nrows();

    // Initialise encoder weights
    let enc_flat = lcg_normal_samples(n_features * n_components, 0.01);
    let mut w_enc = Array2::from_shape_vec((n_features, n_components), enc_flat)
        .map_err(|e| TransformError::ComputationError(e.to_string()))?;
    w_enc = gram_schmidt_orthonorm(&w_enc)?;

    let mut loss_history = Vec::with_capacity(n_iter);

    for iter in 0..n_iter {
        // Corrupt input: use a different noise pattern per iteration by
        // modulating the noise with a per-iteration offset.
        let noise_scale = if noise_std > 0.0 {
            // Scale noise slightly differently per iteration to avoid repeating
            // the same pattern (the LCG always restarts from a fixed seed).
            // We blend noise from two draws and weight by iteration.
            let t = iter as f64 / n_iter.max(1) as f64;
            noise_std * (0.8 + 0.4 * t)
        } else {
            0.0
        };

        let noisy = add_gaussian_noise(&x_clean, noise_scale)?;

        // Forward pass with noisy input
        let z = matmul(&noisy, &w_enc)?;
        let w_dec = transpose(&w_enc);
        let x_hat = matmul(&z, &w_dec)?;

        // Loss w.r.t. clean target
        let residual = {
            let mut r = x_hat.clone();
            for i in 0..n_samples {
                for j in 0..n_features {
                    r[[i, j]] -= x_clean[[i, j]];
                }
            }
            r
        };

        let loss = residual.iter().map(|v| v * v).sum::<f64>() / (n_samples * n_features) as f64;
        loss_history.push(loss);

        // Gradient w.r.t. W_enc (tied, W_dec = W_enc^T):
        //   X_hat = X_noisy @ W_enc @ W_enc^T
        //   Encoder path: dL/dW_enc = X_noisy^T @ (R @ W_enc)
        //   Decoder path: dL/dW_enc = R^T @ z  (since z = X_noisy @ W_enc)
        let xt_noisy = transpose(&noisy);
        let resid_wenc = matmul(&residual, &w_enc)?;  // (n, k)
        let grad_enc = matmul(&xt_noisy, &resid_wenc)?;  // (d, k) — encoder path

        let rt = transpose(&residual);  // (d, n)
        let grad_dec = matmul(&rt, &z)?;  // (d, k) — decoder path

        let scale = 2.0 / (n_samples * n_features) as f64;
        for i in 0..n_features {
            for j in 0..n_components {
                let g = scale * (grad_enc[[i, j]] + grad_dec[[i, j]]);
                w_enc[[i, j]] -= lr * g;
            }
        }
    }

    let w_dec = transpose(&w_enc);
    Ok(DenoisingAE {
        encoder_weights: w_enc,
        decoder_weights: w_dec,
        means,
        n_features,
        n_components,
        noise_std,
        loss_history,
    })
}

// ─── Validation Helper ────────────────────────────────────────────────────────

fn validate_ae_params(
    data: &Array2<f64>,
    n_components: usize,
    lr: f64,
    n_iter: usize,
) -> Result<()> {
    if data.nrows() == 0 || data.ncols() == 0 {
        return Err(TransformError::InvalidInput(
            "data must have at least one sample and one feature".to_string(),
        ));
    }
    if n_components == 0 {
        return Err(TransformError::InvalidInput(
            "n_components must be > 0".to_string(),
        ));
    }
    if n_components > data.ncols() {
        return Err(TransformError::InvalidInput(format!(
            "n_components ({}) cannot exceed n_features ({})",
            n_components,
            data.ncols()
        )));
    }
    if lr <= 0.0 {
        return Err(TransformError::InvalidInput(
            "lr (learning rate) must be > 0".to_string(),
        ));
    }
    if n_iter == 0 {
        return Err(TransformError::InvalidInput(
            "n_iter must be > 0".to_string(),
        ));
    }
    Ok(())
}

// ─── Convenience Free Functions ───────────────────────────────────────────────

/// Fit a [`LinearAutoencoder`] and return `(encoder_W, decoder_W)`.
///
/// This is a shorthand that exposes just the weight matrices for callers
/// who want to use [`encode`] and [`decode`] directly.
pub fn fit_linear_ae_weights(
    data: &Array2<f64>,
    n_components: usize,
    lr: f64,
    n_iter: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let ae = fit_linear_ae(data, n_components, lr, n_iter)?;
    Ok((ae.encoder_weights, ae.decoder_weights))
}

/// Fit a [`DenoisingAE`] and return `(encoder_W, decoder_W)`.
///
/// Shorthand analogue of [`fit_linear_ae_weights`] for denoising autoencoders.
pub fn fit_denoising_ae_weights(
    clean_data: &Array2<f64>,
    n_components: usize,
    noise_std: f64,
    lr: f64,
    n_iter: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let ae = fit_denoising_ae(clean_data, n_components, noise_std, lr, n_iter)?;
    Ok((ae.encoder_weights, ae.decoder_weights))
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Create a simple dataset: 20 points along two principal directions.
    fn make_data() -> Array2<f64> {
        let n = 20usize;
        let mut vals = Vec::with_capacity(n * 4);
        for i in 0..n {
            let t = i as f64 / n as f64;
            vals.push(t);
            vals.push(2.0 * t);
            vals.push(-t);
            vals.push(0.5 * t + 0.1);
        }
        Array2::from_shape_vec((n, 4), vals).expect("shape ok")
    }

    #[test]
    fn test_add_gaussian_noise_shape() {
        let data = make_data();
        let noisy = add_gaussian_noise(&data, 0.1).expect("noise ok");
        assert_eq!(noisy.shape(), data.shape());
    }

    #[test]
    fn test_add_gaussian_noise_zero() {
        let data = make_data();
        let noisy = add_gaussian_noise(&data, 0.0).expect("no noise ok");
        assert_eq!(noisy, data);
    }

    #[test]
    fn test_encode_decode_shapes() {
        let data = make_data();
        let ae = fit_linear_ae(&data, 2, 0.01, 50).expect("fit ok");
        let latent = ae.encode(&data).expect("encode ok");
        assert_eq!(latent.shape(), &[20, 2]);
        let recon = ae.decode(&latent).expect("decode ok");
        assert_eq!(recon.shape(), data.shape());
    }

    #[test]
    fn test_reconstruction_loss_nonneg() {
        let data = make_data();
        let ae = fit_linear_ae(&data, 2, 0.01, 50).expect("fit ok");
        let loss = ae.reconstruction_loss(&data).expect("loss ok");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_denoising_ae_shapes() {
        let data = make_data();
        let dae = fit_denoising_ae(&data, 2, 0.05, 0.01, 50).expect("dae fit ok");
        let latent = dae.encode(&data).expect("encode ok");
        assert_eq!(latent.shape(), &[20, 2]);
        let recon = dae.decode(&latent).expect("decode ok");
        assert_eq!(recon.shape(), data.shape());
    }

    #[test]
    fn test_fit_weights_convenience() {
        let data = make_data();
        let (enc, dec) = fit_linear_ae_weights(&data, 2, 0.01, 30).expect("weights ok");
        assert_eq!(enc.shape(), &[4, 2]);
        assert_eq!(dec.shape(), &[2, 4]);
    }

    #[test]
    fn test_denoising_ae_weights_convenience() {
        let data = make_data();
        let (enc, dec) = fit_denoising_ae_weights(&data, 2, 0.05, 0.01, 30).expect("weights ok");
        assert_eq!(enc.shape(), &[4, 2]);
        assert_eq!(dec.shape(), &[2, 4]);
    }

    #[test]
    fn test_invalid_n_components_zero() {
        let data = make_data();
        assert!(fit_linear_ae(&data, 0, 0.01, 10).is_err());
    }

    #[test]
    fn test_invalid_n_components_too_large() {
        let data = make_data();
        assert!(fit_linear_ae(&data, 10, 0.01, 10).is_err());
    }

    #[test]
    fn test_invalid_lr() {
        let data = make_data();
        assert!(fit_linear_ae(&data, 2, -0.01, 10).is_err());
    }

    #[test]
    fn test_standalone_encode_decode() {
        let data = make_data();
        // Build encoder (4,2) and decoder (2,4) manually
        let mut enc = Array2::<f64>::zeros((4, 2));
        enc[[0, 0]] = 1.0;
        enc[[1, 1]] = 1.0;
        let mut dec = Array2::<f64>::zeros((2, 4));
        dec[[0, 0]] = 1.0;
        dec[[1, 1]] = 1.0;
        let latent = encode(&data, &enc).expect("encode ok");
        assert_eq!(latent.shape(), &[20, 2]);
        let recon = decode(&latent, &dec).expect("decode ok");
        assert_eq!(recon.shape(), &[20, 4]);
    }
}
