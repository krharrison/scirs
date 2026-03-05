//! Linear Autoencoder, Sparse PCA, and Dictionary Learning
//!
//! This module provides struct-based APIs for linear dimensionality reduction
//! via autoencoders and sparsity-constrained representations.
//!
//! ## LinearAutoencoder
//!
//! A linear autoencoder (no activations) is mathematically equivalent to PCA:
//! the encoder is the matrix of top principal components.  Implemented via SVD
//! for numerical stability.
//!
//! ## SparsePca
//!
//! Sparse PCA via alternating minimisation:
//!   min_{Z, D}  ½ ||X − Z D||²_F  +  α ||Z||_1
//!   subject to  ||d_k||_2 ≤ 1  for each atom d_k.
//!
//! ## DictionaryLearning
//!
//! K-SVD dictionary learning: alternating sparse-coding (OMP) and
//! dictionary-atom update via rank-1 SVD, generalising k-means.
//!
//! ## References
//!
//! - Bourlard & Kamp (1988). Auto-association by multilayer perceptrons and SVD.
//! - Olshausen & Field (1997). Sparse coding with an overcomplete basis set.
//! - Aharon, Elad & Bruckstein (2006). K-SVD: An algorithm for designing overcomplete dictionaries.
//! - Mairal et al. (2010). Online learning for matrix factorization and sparse coding.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::Rng;
use scirs2_linalg::svd;

// ─── internal helpers ─────────────────────────────────────────────────────────

/// Compute column means of a data matrix (n_samples × n_features).
fn col_means(data: &Array2<f64>) -> Array1<f64> {
    let n = data.nrows() as f64;
    data.mean_axis(Axis(0)).map_or_else(
        || Array1::zeros(data.ncols()),
        |m| m * (n / n), // identity — just forces the type
    )
}

/// Subtract column means in-place, returning the mean vector.
fn centre(data: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let means = col_means(data);
    let centred = data - &means.view().insert_axis(Axis(0));
    (centred, means)
}

/// Compute ||v||_2.
fn l2_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Soft-threshold: sign(x) * max(|x| - threshold, 0).
fn soft_threshold(x: f64, t: f64) -> f64 {
    if x > t {
        x - t
    } else if x < -t {
        x + t
    } else {
        0.0
    }
}

/// Lasso coordinate descent for a single sample:
///   min_{z}  ½ ||x − D z||²  +  α ||z||_1
///
/// D has shape (n_atoms, n_features), x has length n_features.
/// Returns z of length n_atoms.
fn lasso_cd_single(x: &Array1<f64>, dict: &Array2<f64>, alpha: f64, max_iter: usize, tol: f64) -> Array1<f64> {
    let n_atoms = dict.nrows();
    let mut z = Array1::<f64>::zeros(n_atoms);

    // Gram matrix G = D D^T and correlation c = D x
    let mut gram = Array2::<f64>::zeros((n_atoms, n_atoms));
    let mut corr = Array1::<f64>::zeros(n_atoms);
    for i in 0..n_atoms {
        for k in 0..x.len() {
            corr[i] += dict[[i, k]] * x[k];
        }
        for j in 0..n_atoms {
            for k in 0..x.len() {
                gram[[i, j]] += dict[[i, k]] * dict[[j, k]];
            }
        }
    }

    for _iter in 0..max_iter {
        let z_prev = z.clone();
        for i in 0..n_atoms {
            // Partial residual: r_i = corr[i] - sum_{j≠i} G[i,j] * z[j]
            let mut r_i = corr[i];
            for j in 0..n_atoms {
                if j != i {
                    r_i -= gram[[i, j]] * z[j];
                }
            }
            let denom = gram[[i, i]];
            if denom.abs() > 1e-12 {
                z[i] = soft_threshold(r_i / denom, alpha / denom);
            } else {
                z[i] = 0.0;
            }
        }
        // Convergence check
        let diff: f64 = z.iter().zip(z_prev.iter()).map(|(a, b)| (a - b).abs()).sum();
        if diff < tol {
            break;
        }
    }
    z
}

/// Orthogonal Matching Pursuit for a single sample:
///   min_{z}  ||x − D^T z||²  subject to  ||z||_0 ≤ n_nonzero
///
/// D has shape (n_atoms, n_features).
/// Returns z of length n_atoms.
fn omp_single(x: &Array1<f64>, dict: &Array2<f64>, n_nonzero: usize) -> Array1<f64> {
    let n_atoms = dict.nrows();
    let n_feat = x.len();
    let n_nonzero = n_nonzero.min(n_atoms);

    let mut residual = x.clone();
    let mut support: Vec<usize> = Vec::with_capacity(n_nonzero);
    let mut z = Array1::<f64>::zeros(n_atoms);

    for _ in 0..n_nonzero {
        // Select atom most correlated with residual
        let mut best_idx = 0;
        let mut best_corr: f64 = f64::NEG_INFINITY;
        for j in 0..n_atoms {
            if support.contains(&j) {
                continue;
            }
            let c: f64 = dict.row(j).iter().zip(residual.iter()).map(|(d, r)| d * r).sum::<f64>().abs();
            if c > best_corr {
                best_corr = c;
                best_idx = j;
            }
        }
        support.push(best_idx);

        // Least squares on support: min ||x − D_S c||²
        let s = support.len();
        let mut ds = Array2::<f64>::zeros((s, n_feat));
        for (row, &idx) in support.iter().enumerate() {
            ds.row_mut(row).assign(&dict.row(idx));
        }
        // Normal equations: (D_S D_S^T) c = D_S x
        let mut gram = Array2::<f64>::zeros((s, s));
        let mut rhs = Array1::<f64>::zeros(s);
        for i in 0..s {
            rhs[i] = ds.row(i).iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            for j in 0..s {
                gram[[i, j]] = ds.row(i).iter().zip(ds.row(j).iter()).map(|(a, b)| a * b).sum();
            }
        }
        // Solve via Cholesky-like: simple Gaussian elimination for small s
        let coeffs = solve_normal_small(&gram, &rhs);

        // Update residual
        residual = x.clone();
        for (i, &idx) in support.iter().enumerate() {
            for f in 0..n_feat {
                residual[f] -= coeffs[i] * dict[[idx, f]];
            }
        }

        // Store solution
        z.fill(0.0);
        for (i, &idx) in support.iter().enumerate() {
            z[idx] = coeffs[i];
        }
    }
    z
}

/// Small dense linear solve via Gaussian elimination with partial pivoting.
fn solve_normal_small(a: &Array2<f64>, b: &Array1<f64>) -> Vec<f64> {
    let n = a.nrows();
    if n == 0 {
        return vec![];
    }
    // Augmented matrix [A | b]
    let mut aug: Vec<f64> = Vec::with_capacity(n * (n + 1));
    for i in 0..n {
        for j in 0..n {
            aug.push(a[[i, j]]);
        }
        aug.push(b[i]);
    }

    let stride = n + 1;
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * stride + col].abs();
        let mut pivot_row = col;
        for row in (col + 1)..n {
            let v = aug[row * stride + col].abs();
            if v > max_val {
                max_val = v;
                pivot_row = row;
            }
        }
        if max_val < 1e-14 {
            continue; // singular / near-singular
        }
        // Swap rows
        if pivot_row != col {
            for k in 0..=n {
                aug.swap(col * stride + k, pivot_row * stride + k);
            }
        }
        let diag = aug[col * stride + col];
        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row * stride + col] / diag;
            for k in col..=n {
                let sub = factor * aug[col * stride + k];
                aug[row * stride + k] -= sub;
            }
        }
    }
    // Back-substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * stride + n];
        for j in (i + 1)..n {
            sum -= aug[i * stride + j] * x[j];
        }
        let d = aug[i * stride + i];
        x[i] = if d.abs() < 1e-14 { 0.0 } else { sum / d };
    }
    x
}

// ─── LinearAutoencoder ────────────────────────────────────────────────────────

/// Linear Autoencoder backed by SVD (equivalent to PCA).
///
/// The encoder weight matrix W_enc ∈ ℝ^{k×d} is the top-k left singular
/// vectors of the centred data matrix.  The decoder W_dec = W_enc^T.
///
/// Encoding:   z = (x − μ) W_enc^T
/// Decoding:   x̂ = z W_dec^T + μ
/// Reconstruct: x̂ = encode then decode
///
/// ## Example
///
/// ```rust
/// use scirs2_transform::linear_ae::LinearAutoencoder;
///
/// let data = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![4.0, 5.0, 6.0],
///     vec![7.0, 8.0, 9.0],
///     vec![2.0, 3.0, 4.0],
/// ];
/// let mut ae = LinearAutoencoder::new(2);
/// ae.fit(&data).expect("should succeed");
/// let encoded = ae.encode(&data);
/// assert_eq!(encoded.len(), 4);
/// assert_eq!(encoded[0].len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct LinearAutoencoder {
    /// Encoder weight matrix, shape (n_components × n_features)
    pub encoder: Vec<Vec<f64>>,
    /// Decoder weight matrix, shape (n_features × n_components)
    pub decoder: Vec<Vec<f64>>,
    /// Number of latent dimensions
    pub n_components: usize,
    /// Per-feature mean computed during fit
    mean: Vec<f64>,
    /// Singular values (explained variance)
    singular_values: Vec<f64>,
}

impl LinearAutoencoder {
    /// Create a new (unfitted) LinearAutoencoder.
    ///
    /// # Arguments
    /// * `n_components` — number of latent dimensions (must be ≥ 1)
    pub fn new(n_components: usize) -> Self {
        Self {
            encoder: Vec::new(),
            decoder: Vec::new(),
            n_components,
            mean: Vec::new(),
            singular_values: Vec::new(),
        }
    }

    /// Fit the autoencoder to data using SVD.
    ///
    /// # Arguments
    /// * `data` — slice of row vectors, each of length n_features
    pub fn fit(&mut self, data: &[Vec<f64>]) -> Result<()> {
        let n_samples = data.len();
        if n_samples == 0 {
            return Err(TransformError::InvalidInput("Empty data".to_string()));
        }
        let n_features = data[0].len();
        if n_features == 0 {
            return Err(TransformError::InvalidInput("Zero-length feature vectors".to_string()));
        }
        if self.n_components == 0 || self.n_components > n_features.min(n_samples) {
            return Err(TransformError::InvalidInput(format!(
                "n_components ({}) must be in 1..=min(n_samples, n_features) = {}",
                self.n_components,
                n_features.min(n_samples)
            )));
        }

        // Build ndarray matrix from row vecs
        let flat: Vec<f64> = data.iter().flat_map(|row| row.iter().copied()).collect();
        let x = Array2::from_shape_vec((n_samples, n_features), flat)
            .map_err(|e| TransformError::ComputationError(e.to_string()))?;

        // Centre data
        let (xc, means) = centre(&x);
        self.mean = means.to_vec();

        // SVD: xc = U S V^T  (V columns are right singular vectors)
        let (u_opt, s, vt) = svd(&xc, true, false)
            .map_err(|e| TransformError::ComputationError(format!("SVD failed: {e}")))?;
        let _u = u_opt.ok_or_else(|| TransformError::ComputationError("SVD returned no U".to_string()))?;

        // Top-k rows of Vt are the principal directions
        let k = self.n_components;
        let n_sv = s.len().min(k);
        self.singular_values = s.iter().take(n_sv).copied().collect();

        // encoder[i][j] = Vt[i, j]  (i = component, j = feature)
        self.encoder = (0..k)
            .map(|i| {
                if i < vt.nrows() {
                    vt.row(i).to_vec()
                } else {
                    vec![0.0; n_features]
                }
            })
            .collect();

        // decoder = encoder transposed:  decoder[j][i] = encoder[i][j]
        self.decoder = (0..n_features)
            .map(|j| {
                (0..k)
                    .map(|i| self.encoder[i][j])
                    .collect()
            })
            .collect();

        Ok(())
    }

    /// Project data to latent space.
    ///
    /// # Arguments
    /// * `data` — slice of row vectors, each of length n_features
    pub fn encode(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let k = self.n_components;
        data.iter()
            .map(|row| {
                let centred: Vec<f64> = row.iter().enumerate().map(|(j, &v)| v - self.mean.get(j).copied().unwrap_or(0.0)).collect();
                (0..k)
                    .map(|i| {
                        centred.iter().enumerate().map(|(j, &x)| x * self.encoder[i][j]).sum()
                    })
                    .collect()
            })
            .collect()
    }

    /// Reconstruct from latent codes back to feature space.
    ///
    /// # Arguments
    /// * `codes` — slice of code vectors, each of length n_components
    pub fn decode(&self, codes: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_features = self.mean.len();
        let k = self.n_components;
        codes.iter()
            .map(|z| {
                (0..n_features)
                    .map(|j| {
                        let rec: f64 = (0..k).map(|i| z.get(i).copied().unwrap_or(0.0) * self.decoder[j][i]).sum();
                        rec + self.mean.get(j).copied().unwrap_or(0.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// Encode then decode (full round-trip reconstruction).
    pub fn reconstruct(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let codes = self.encode(data);
        self.decode(&codes)
    }

    /// Mean squared reconstruction error over the dataset.
    pub fn reconstruction_error(&self, data: &[Vec<f64>]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let reconstructed = self.reconstruct(data);
        let total: f64 = data.iter()
            .zip(reconstructed.iter())
            .flat_map(|(orig, rec)| {
                orig.iter().zip(rec.iter()).map(|(a, b)| (a - b).powi(2))
            })
            .sum();
        total / (data.len() * data[0].len().max(1)) as f64
    }

    /// Singular values from the fit (proportional to explained variance).
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    /// Explained variance ratio for each component.
    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        let total: f64 = self.singular_values.iter().map(|s| s * s).sum::<f64>();
        if total < 1e-14 {
            return vec![0.0; self.singular_values.len()];
        }
        self.singular_values.iter().map(|s| s * s / total).collect()
    }
}

// ─── SparsePca (struct API) ───────────────────────────────────────────────────

/// Sparse PCA via alternating minimisation with L1 regularisation.
///
/// Solves:
///   min_{Z, D}  ½ ||X − Z D||²_F  +  α ||Z||_1
///   s.t.        ||d_k||_2 ≤ 1  for each atom d_k.
///
/// The dictionary D has shape (n_components × n_features).
/// The codes Z have shape (n_samples × n_components).
///
/// ## Example
///
/// ```rust
/// use scirs2_transform::linear_ae::SparsePca;
///
/// let data = vec![
///     vec![1.0, 2.0, 3.0, 4.0],
///     vec![5.0, 6.0, 7.0, 8.0],
///     vec![9.0, 8.0, 7.0, 6.0],
///     vec![2.0, 4.0, 6.0, 8.0],
///     vec![1.0, 3.0, 5.0, 7.0],
/// ];
/// let mut spca = SparsePca::new(2, 0.5);
/// spca.fit(&data, 50, 1e-4).expect("should succeed");
/// let codes = spca.transform(&data);
/// assert_eq!(codes.len(), 5);
/// assert_eq!(codes[0].len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct SparsePca {
    /// Sparse components / dictionary, shape (n_components × n_features)
    pub components: Vec<Vec<f64>>,
    /// Number of sparse components
    pub n_components: usize,
    /// L1 regularisation strength
    pub alpha: f64,
    /// Per-feature mean (for centring)
    mean: Vec<f64>,
    /// Number of alternating iterations run during fit
    n_iter_done: usize,
}

impl SparsePca {
    /// Create a new (unfitted) SparsePca.
    ///
    /// # Arguments
    /// * `n_components` — number of sparse components
    /// * `alpha` — L1 regularisation weight (larger → sparser codes)
    pub fn new(n_components: usize, alpha: f64) -> Self {
        Self {
            components: Vec::new(),
            n_components,
            alpha,
            mean: Vec::new(),
            n_iter_done: 0,
        }
    }

    /// Fit SparsePca to data.
    ///
    /// # Arguments
    /// * `data`     — input data, n_samples rows × n_features cols
    /// * `max_iter` — maximum number of alternating iterations
    /// * `tol`      — convergence tolerance on objective change
    pub fn fit(&mut self, data: &[Vec<f64>], max_iter: usize, tol: f64) -> Result<()> {
        let n_samples = data.len();
        if n_samples == 0 {
            return Err(TransformError::InvalidInput("Empty data".to_string()));
        }
        let n_features = data[0].len();
        if n_features == 0 {
            return Err(TransformError::InvalidInput("Zero-length feature vectors".to_string()));
        }
        let k = self.n_components;
        if k == 0 || k > n_features {
            return Err(TransformError::InvalidInput(format!(
                "n_components ({k}) must be in 1..={n_features}"
            )));
        }
        if self.alpha < 0.0 {
            return Err(TransformError::InvalidInput("alpha must be >= 0".to_string()));
        }

        // Build ndarray data matrix and centre it
        let flat: Vec<f64> = data.iter().flat_map(|row| row.iter().copied()).collect();
        let x = Array2::from_shape_vec((n_samples, n_features), flat)
            .map_err(|e| TransformError::ComputationError(e.to_string()))?;
        let (xc, means) = centre(&x);
        self.mean = means.to_vec();

        // Initialise dictionary from top-k right singular vectors
        let mut rng = scirs2_core::random::rng();
        let (_u_opt, _s, vt) = svd(&xc, true, false)
            .map_err(|e| TransformError::ComputationError(format!("SVD init failed: {e}")))?;

        let mut dict = Array2::<f64>::zeros((k, n_features));
        for i in 0..k {
            if i < vt.nrows() {
                dict.row_mut(i).assign(&vt.row(i));
            } else {
                // random fallback
                for j in 0..n_features {
                    dict[[i, j]] = rng.random_range(-0.1..0.1_f64);
                }
                // normalise
                let norm = l2_norm(&dict.row(i).to_owned());
                if norm > 1e-10 {
                    let row = dict.row(i).to_owned() / norm;
                    dict.row_mut(i).assign(&row);
                }
            }
        }

        // Alternating minimisation
        let lasso_max_iter = 200_usize;
        let lasso_tol = 1e-6_f64;
        let mut codes = Array2::<f64>::zeros((n_samples, k));
        let mut prev_obj = f64::INFINITY;

        for iter in 0..max_iter {
            // Code step: update Z row-by-row via Lasso CD
            for s_idx in 0..n_samples {
                let x_row = xc.row(s_idx).to_owned();
                let z = lasso_cd_single(&x_row, &dict, self.alpha, lasso_max_iter, lasso_tol);
                codes.row_mut(s_idx).assign(&z);
            }

            // Dictionary step: update each atom d_i to minimise residual
            // Closed form: d_i = R_i z_i / ||R_i z_i||, where R_i is
            // residual with d_i contribution removed.
            for i in 0..k {
                // Residual excluding atom i
                // R = X - Z_{:, !=i} D_{!=i, :}  (contribution from atom i column)
                let mut r = xc.clone();
                for j in 0..k {
                    if j != i {
                        // Subtract outer product: codes[:, j] ⊗ dict[j, :]
                        for s_idx in 0..n_samples {
                            let c_sj = codes[[s_idx, j]];
                            for f in 0..n_features {
                                r[[s_idx, f]] -= c_sj * dict[[j, f]];
                            }
                        }
                    }
                }
                // Codes for atom i: col vector z_i ∈ ℝ^{n_samples}
                let z_i = codes.column(i).to_owned();
                // New atom: d_i_new = R^T z_i / ||z_i||²
                let z_norm_sq: f64 = z_i.iter().map(|v| v * v).sum();
                let mut new_atom = Array1::<f64>::zeros(n_features);
                if z_norm_sq > 1e-14 {
                    for s_idx in 0..n_samples {
                        for f in 0..n_features {
                            new_atom[f] += z_i[s_idx] * r[[s_idx, f]];
                        }
                    }
                    let s = new_atom.iter().copied();
                    let scale = 1.0 / z_norm_sq;
                    for f in 0..n_features {
                        new_atom[f] *= scale;
                    }
                    // Normalise to unit norm
                    let norm = l2_norm(&new_atom);
                    if norm > 1e-10 {
                        let _ = s; // suppress unused warning
                        for f in 0..n_features {
                            new_atom[f] /= norm;
                        }
                    }
                } else {
                    // Atom unused: reinitialise randomly
                    for f in 0..n_features {
                        new_atom[f] = rng.random_range(-0.1..0.1_f64);
                    }
                    let norm = l2_norm(&new_atom);
                    if norm > 1e-10 {
                        for f in 0..n_features {
                            new_atom[f] /= norm;
                        }
                    }
                }
                dict.row_mut(i).assign(&new_atom);
            }

            // Objective: ½ ||X - Z D||² + α ||Z||_1
            let residual = &xc - codes.dot(&dict);
            let frob_sq: f64 = residual.iter().map(|v| v * v).sum();
            let l1_z: f64 = codes.iter().map(|v| v.abs()).sum();
            let obj = 0.5 * frob_sq + self.alpha * l1_z;

            if (prev_obj - obj).abs() < tol * (1.0 + prev_obj.abs()) {
                self.n_iter_done = iter + 1;
                break;
            }
            prev_obj = obj;
            self.n_iter_done = iter + 1;
        }

        // Store dictionary as Vec<Vec<f64>>
        self.components = (0..k)
            .map(|i| dict.row(i).to_vec())
            .collect();

        Ok(())
    }

    /// Transform data to sparse code representation.
    ///
    /// Applies Lasso coordinate descent per sample using the fitted dictionary.
    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if self.components.is_empty() || data.is_empty() {
            return vec![vec![0.0; self.n_components]; data.len()];
        }
        let n_features = self.mean.len();
        let k = self.n_components;

        // Build dict Array2 from components
        let flat: Vec<f64> = self.components.iter().flat_map(|row| row.iter().copied()).collect();
        let dict = match Array2::from_shape_vec((k, n_features), flat) {
            Ok(d) => d,
            Err(_) => return vec![vec![0.0; k]; data.len()],
        };

        let lasso_max_iter = 200_usize;
        let lasso_tol = 1e-6_f64;

        data.iter()
            .map(|row| {
                let centred: Array1<f64> = Array1::from_iter(
                    row.iter().enumerate().map(|(j, &v)| v - self.mean.get(j).copied().unwrap_or(0.0))
                );
                lasso_cd_single(&centred, &dict, self.alpha, lasso_max_iter, lasso_tol).to_vec()
            })
            .collect()
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, data: &[Vec<f64>], max_iter: usize, tol: f64) -> Vec<Vec<f64>> {
        match self.fit(data, max_iter, tol) {
            Ok(()) => self.transform(data),
            Err(_) => vec![vec![0.0; self.n_components]; data.len()],
        }
    }

    /// Number of alternating iterations completed during fit.
    pub fn n_iter_done(&self) -> usize {
        self.n_iter_done
    }

    /// Compute reconstruction error on data.
    pub fn reconstruction_error(&self, data: &[Vec<f64>]) -> f64 {
        if self.components.is_empty() || data.is_empty() {
            return 0.0;
        }
        let n_features = self.mean.len();
        let k = self.n_components;
        let flat: Vec<f64> = self.components.iter().flat_map(|r| r.iter().copied()).collect();
        let dict = match Array2::from_shape_vec((k, n_features), flat) {
            Ok(d) => d,
            Err(_) => return f64::NAN,
        };

        let codes = self.transform(data);
        let mut total_sq = 0.0_f64;
        let mut count = 0_usize;
        for (row, code) in data.iter().zip(codes.iter()) {
            for j in 0..n_features {
                let centred = row.get(j).copied().unwrap_or(0.0) - self.mean.get(j).copied().unwrap_or(0.0);
                let rec: f64 = (0..k).map(|i| code.get(i).copied().unwrap_or(0.0) * dict[[i, j]]).sum();
                total_sq += (centred - rec).powi(2);
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { total_sq / count as f64 }
    }
}

// ─── DictionaryLearning (struct API) ─────────────────────────────────────────

/// K-SVD Dictionary Learning.
///
/// Alternates between:
/// 1. **Sparse coding** each sample using OMP (n_nonzero nonzeros per code).
/// 2. **Atom update** via rank-1 SVD of the restricted residual matrix.
///
/// ## Example
///
/// ```rust
/// use scirs2_transform::linear_ae::DictionaryLearning;
///
/// let data = vec![
///     vec![1.0, 0.0, 3.0],
///     vec![0.0, 2.0, 1.0],
///     vec![2.0, 1.0, 0.0],
///     vec![1.0, 2.0, 3.0],
///     vec![3.0, 1.0, 2.0],
/// ];
/// let mut dl = DictionaryLearning::new(3, 1.0);
/// dl.fit(&data, 20).expect("should succeed");
/// let codes = dl.transform(&data);
/// assert_eq!(codes.len(), 5);
/// assert_eq!(codes[0].len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct DictionaryLearning {
    /// Dictionary atoms, shape (n_atoms × n_features)
    pub dictionary: Vec<Vec<f64>>,
    /// Number of atoms in the dictionary
    pub n_atoms: usize,
    /// OMP sparsity parameter (max non-zeros per code vector)
    pub alpha: f64,
    /// Per-feature mean (centring)
    mean: Vec<f64>,
    /// Number of iterations performed
    n_iter_done: usize,
}

impl DictionaryLearning {
    /// Create a new (unfitted) DictionaryLearning.
    ///
    /// # Arguments
    /// * `n_atoms` — number of dictionary atoms (can be > n_features for overcomplete)
    /// * `alpha`   — OMP sparsity (number of nonzeros per code, ≥ 1)
    pub fn new(n_atoms: usize, alpha: f64) -> Self {
        Self {
            dictionary: Vec::new(),
            n_atoms,
            alpha,
            mean: Vec::new(),
            n_iter_done: 0,
        }
    }

    /// Fit the dictionary to data using K-SVD.
    ///
    /// # Arguments
    /// * `data`   — n_samples × n_features input
    /// * `n_iter` — number of K-SVD iterations
    pub fn fit(&mut self, data: &[Vec<f64>], n_iter: usize) -> Result<()> {
        let n_samples = data.len();
        if n_samples == 0 {
            return Err(TransformError::InvalidInput("Empty data".to_string()));
        }
        let n_features = data[0].len();
        if n_features == 0 {
            return Err(TransformError::InvalidInput("Zero-length feature vectors".to_string()));
        }
        let k = self.n_atoms;
        if k == 0 {
            return Err(TransformError::InvalidInput("n_atoms must be >= 1".to_string()));
        }

        // Build data matrix and centre
        let flat: Vec<f64> = data.iter().flat_map(|row| row.iter().copied()).collect();
        let x = Array2::from_shape_vec((n_samples, n_features), flat)
            .map_err(|e| TransformError::ComputationError(e.to_string()))?;
        let (xc, means) = centre(&x);
        self.mean = means.to_vec();

        // Initialise: take k random samples as initial atoms (with normalisation)
        let mut rng = scirs2_core::random::rng();
        let mut dict = Array2::<f64>::zeros((k, n_features));
        for i in 0..k {
            let s_idx = (rng.next_u64() as usize) % n_samples;
            let row = xc.row(s_idx).to_owned();
            let norm = l2_norm(&row);
            if norm > 1e-10 {
                dict.row_mut(i).assign(&(row / norm));
            } else {
                for j in 0..n_features {
                    dict[[i, j]] = rng.random_range(-0.1..0.1_f64);
                }
                let r2 = l2_norm(&dict.row(i).to_owned());
                if r2 > 1e-10 {
                    let tmp = dict.row(i).to_owned() / r2;
                    dict.row_mut(i).assign(&tmp);
                }
            }
        }

        let n_nonzero = (self.alpha.round() as usize).max(1).min(k);

        for iter in 0..n_iter {
            // Sparse coding step: encode all samples via OMP
            let mut codes = Array2::<f64>::zeros((n_samples, k));
            for s_idx in 0..n_samples {
                let x_row = xc.row(s_idx).to_owned();
                let z = omp_single(&x_row, &dict, n_nonzero);
                codes.row_mut(s_idx).assign(&z);
            }

            // Dictionary update step: for each atom j, update via rank-1 SVD
            for j in 0..k {
                // Find samples using atom j
                let users: Vec<usize> = (0..n_samples)
                    .filter(|&s_idx| codes[[s_idx, j]].abs() > 1e-10)
                    .collect();

                if users.is_empty() {
                    // Atom unused: reinitialise from worst-reconstructed sample
                    let errors: Vec<f64> = (0..n_samples)
                        .map(|s_idx| {
                            let rec = codes.row(s_idx).iter().enumerate()
                                .map(|(i, &c)| c * dict[[i, 0]])
                                .sum::<f64>();
                            let _ = rec;
                            // squared error for this sample
                            let mut e2 = 0.0_f64;
                            for f in 0..n_features {
                                let r: f64 = codes.row(s_idx).iter().enumerate()
                                    .map(|(ai, &c)| c * dict[[ai, f]])
                                    .sum();
                                e2 += (xc[[s_idx, f]] - r).powi(2);
                            }
                            e2
                        })
                        .collect();
                    let worst = errors.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let row = xc.row(worst).to_owned();
                    let norm = l2_norm(&row);
                    if norm > 1e-10 {
                        dict.row_mut(j).assign(&(row / norm));
                    }
                    continue;
                }

                // Restricted residual: E_j = X_users - Z_users_{!=j} * D_{!=j}
                let n_users = users.len();
                let mut ej = Array2::<f64>::zeros((n_users, n_features));
                for (ui, &s_idx) in users.iter().enumerate() {
                    for f in 0..n_features {
                        let mut r = xc[[s_idx, f]];
                        for ai in 0..k {
                            if ai != j {
                                r -= codes[[s_idx, ai]] * dict[[ai, f]];
                            }
                        }
                        ej[[ui, f]] = r;
                    }
                }

                // Rank-1 SVD of E_j: E_j ≈ sigma * u * v^T
                // d_j_new = v (first right singular vector)
                // Update codes: z[users, j] = sigma * u
                let (u_opt, s, vt) = svd(&ej, true, false)
                    .map_err(|e| TransformError::ComputationError(format!("K-SVD inner SVD: {e}")))?;

                if let Some(u_mat) = u_opt {
                    let sigma = if !s.is_empty() { s[0] } else { 0.0 };
                    // New atom = first row of Vt
                    if vt.nrows() > 0 {
                        dict.row_mut(j).assign(&vt.row(0));
                    }
                    // Update codes for users: z[s_idx, j] = sigma * u[ui, 0]
                    for (ui, &s_idx) in users.iter().enumerate() {
                        if u_mat.ncols() > 0 {
                            codes[[s_idx, j]] = sigma * u_mat[[ui, 0]];
                        }
                    }
                }
            }

            self.n_iter_done = iter + 1;
        }

        self.dictionary = (0..k).map(|i| dict.row(i).to_vec()).collect();
        Ok(())
    }

    /// Encode data using the fitted dictionary via OMP.
    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if self.dictionary.is_empty() || data.is_empty() {
            return vec![vec![0.0; self.n_atoms]; data.len()];
        }
        let n_features = self.mean.len();
        let k = self.n_atoms;
        let flat: Vec<f64> = self.dictionary.iter().flat_map(|r| r.iter().copied()).collect();
        let dict = match Array2::from_shape_vec((k, n_features), flat) {
            Ok(d) => d,
            Err(_) => return vec![vec![0.0; k]; data.len()],
        };
        let n_nonzero = (self.alpha.round() as usize).max(1).min(k);
        data.iter()
            .map(|row| {
                let centred: Array1<f64> = Array1::from_iter(
                    row.iter().enumerate().map(|(j, &v)| v - self.mean.get(j).copied().unwrap_or(0.0))
                );
                omp_single(&centred, &dict, n_nonzero).to_vec()
            })
            .collect()
    }

    /// Fit dictionary and return codes for the training data.
    pub fn fit_transform(&mut self, data: &[Vec<f64>], n_iter: usize) -> Vec<Vec<f64>> {
        match self.fit(data, n_iter) {
            Ok(()) => self.transform(data),
            Err(_) => vec![vec![0.0; self.n_atoms]; data.len()],
        }
    }

    /// Number of K-SVD iterations completed.
    pub fn n_iter_done(&self) -> usize {
        self.n_iter_done
    }

    /// Compute mean squared reconstruction error.
    pub fn reconstruction_error(&self, data: &[Vec<f64>]) -> f64 {
        if self.dictionary.is_empty() || data.is_empty() {
            return 0.0;
        }
        let n_features = self.mean.len();
        let k = self.n_atoms;
        let flat: Vec<f64> = self.dictionary.iter().flat_map(|r| r.iter().copied()).collect();
        let dict = match Array2::from_shape_vec((k, n_features), flat) {
            Ok(d) => d,
            Err(_) => return f64::NAN,
        };
        let codes = self.transform(data);
        let mut total_sq = 0.0_f64;
        let mut count = 0_usize;
        for (row, code) in data.iter().zip(codes.iter()) {
            for j in 0..n_features {
                let centred = row.get(j).copied().unwrap_or(0.0) - self.mean.get(j).copied().unwrap_or(0.0);
                let rec: f64 = (0..k).map(|i| code.get(i).copied().unwrap_or(0.0) * dict[[i, j]]).sum();
                total_sq += (centred - rec).powi(2);
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { total_sq / count as f64 }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, d: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..d).map(|j| (i * d + j) as f64 * 0.1).collect())
            .collect()
    }

    // ── LinearAutoencoder tests ───────────────────────────────────────────────

    #[test]
    fn test_linear_ae_fit_shape() {
        let data = make_data(10, 5);
        let mut ae = LinearAutoencoder::new(3);
        ae.fit(&data).expect("fit");
        assert_eq!(ae.encoder.len(), 3);
        assert_eq!(ae.encoder[0].len(), 5);
        assert_eq!(ae.decoder.len(), 5);
        assert_eq!(ae.decoder[0].len(), 3);
    }

    #[test]
    fn test_linear_ae_encode_decode_shape() {
        let data = make_data(8, 4);
        let mut ae = LinearAutoencoder::new(2);
        ae.fit(&data).expect("fit");
        let codes = ae.encode(&data);
        assert_eq!(codes.len(), 8);
        assert_eq!(codes[0].len(), 2);
        let reconstructed = ae.decode(&codes);
        assert_eq!(reconstructed.len(), 8);
        assert_eq!(reconstructed[0].len(), 4);
    }

    #[test]
    fn test_linear_ae_reconstruct_all_components() {
        // With n_components = n_features, reconstruction should be near-perfect
        let data = make_data(5, 3);
        let mut ae = LinearAutoencoder::new(3);
        ae.fit(&data).expect("fit");
        let err = ae.reconstruction_error(&data);
        assert!(
            err < 1e-10,
            "Full-rank AE should reconstruct perfectly, error = {err}"
        );
    }

    #[test]
    fn test_linear_ae_reconstruct_partial() {
        let data = make_data(10, 5);
        let mut ae = LinearAutoencoder::new(2);
        ae.fit(&data).expect("fit");
        let err = ae.reconstruction_error(&data);
        assert!(err.is_finite(), "reconstruction error must be finite");
        assert!(err >= 0.0);
    }

    #[test]
    fn test_linear_ae_singular_values_sorted() {
        let data = make_data(12, 4);
        let mut ae = LinearAutoencoder::new(4);
        ae.fit(&data).expect("fit");
        let sv = ae.singular_values();
        for w in sv.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-10,
                "singular values must be non-increasing: {:.4} < {:.4}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_linear_ae_explained_variance_ratio_sums_one() {
        let data = make_data(8, 4);
        let mut ae = LinearAutoencoder::new(4);
        ae.fit(&data).expect("fit");
        let evr = ae.explained_variance_ratio();
        let sum: f64 = evr.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-8,
            "EVR should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_linear_ae_too_many_components_error() {
        let data = make_data(3, 4);
        let mut ae = LinearAutoencoder::new(5); // > min(n_samples, n_features)
        assert!(ae.fit(&data).is_err());
    }

    #[test]
    fn test_linear_ae_empty_data_error() {
        let data: Vec<Vec<f64>> = vec![];
        let mut ae = LinearAutoencoder::new(2);
        assert!(ae.fit(&data).is_err());
    }

    // ── SparsePca tests ───────────────────────────────────────────────────────

    #[test]
    fn test_sparse_pca_fit_transform_shape() {
        let data = make_data(10, 6);
        let mut spca = SparsePca::new(3, 0.1);
        let codes = spca.fit_transform(&data, 30, 1e-5);
        assert_eq!(codes.len(), 10);
        assert_eq!(codes[0].len(), 3);
    }

    #[test]
    fn test_sparse_pca_components_unit_norm() {
        let data = make_data(8, 4);
        let mut spca = SparsePca::new(2, 0.5);
        spca.fit(&data, 30, 1e-5).expect("fit");
        for comp in &spca.components {
            let norm: f64 = comp.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.1 || norm < 1e-8,
                "component norm should be ~1, got {norm}"
            );
        }
    }

    #[test]
    fn test_sparse_pca_error_invalid_components() {
        let data = make_data(5, 3);
        let mut spca = SparsePca::new(5, 0.1); // > n_features
        assert!(spca.fit(&data, 10, 1e-4).is_err());
    }

    #[test]
    fn test_sparse_pca_transform_separate() {
        let train = make_data(10, 4);
        let test = make_data(4, 4);
        let mut spca = SparsePca::new(2, 0.2);
        spca.fit(&train, 20, 1e-5).expect("fit");
        let codes = spca.transform(&test);
        assert_eq!(codes.len(), 4);
        assert_eq!(codes[0].len(), 2);
    }

    #[test]
    fn test_sparse_pca_n_iter_done() {
        let data = make_data(8, 4);
        let mut spca = SparsePca::new(2, 0.3);
        spca.fit(&data, 50, 1e-6).expect("fit");
        assert!(spca.n_iter_done() > 0);
        assert!(spca.n_iter_done() <= 50);
    }

    #[test]
    fn test_sparse_pca_reconstruction_error_finite() {
        let data = make_data(8, 4);
        let mut spca = SparsePca::new(2, 0.2);
        spca.fit(&data, 30, 1e-5).expect("fit");
        let err = spca.reconstruction_error(&data);
        assert!(err.is_finite());
        assert!(err >= 0.0);
    }

    // ── DictionaryLearning tests ──────────────────────────────────────────────

    #[test]
    fn test_dict_learning_fit_shape() {
        let data = make_data(10, 5);
        let mut dl = DictionaryLearning::new(4, 2.0);
        dl.fit(&data, 20).expect("fit");
        assert_eq!(dl.dictionary.len(), 4);
        assert_eq!(dl.dictionary[0].len(), 5);
    }

    #[test]
    fn test_dict_learning_transform_shape() {
        let data = make_data(8, 4);
        let mut dl = DictionaryLearning::new(6, 2.0);
        dl.fit(&data, 15).expect("fit");
        let codes = dl.transform(&data);
        assert_eq!(codes.len(), 8);
        assert_eq!(codes[0].len(), 6);
    }

    #[test]
    fn test_dict_learning_fit_transform() {
        let data = make_data(10, 4);
        let mut dl = DictionaryLearning::new(4, 2.0);
        let codes = dl.fit_transform(&data, 20);
        assert_eq!(codes.len(), 10);
        assert_eq!(codes[0].len(), 4);
    }

    #[test]
    fn test_dict_learning_n_iter_done() {
        let data = make_data(8, 3);
        let mut dl = DictionaryLearning::new(3, 1.0);
        dl.fit(&data, 30).expect("fit");
        assert_eq!(dl.n_iter_done(), 30);
    }

    #[test]
    fn test_dict_learning_reconstruction_error_finite() {
        let data = make_data(8, 4);
        let mut dl = DictionaryLearning::new(4, 2.0);
        dl.fit(&data, 20).expect("fit");
        let err = dl.reconstruction_error(&data);
        assert!(err.is_finite());
        assert!(err >= 0.0);
    }

    #[test]
    fn test_dict_learning_empty_error() {
        let data: Vec<Vec<f64>> = vec![];
        let mut dl = DictionaryLearning::new(3, 1.0);
        assert!(dl.fit(&data, 10).is_err());
    }

    #[test]
    fn test_dict_learning_overcomplete() {
        // n_atoms > n_features (overcomplete dictionary)
        let data = make_data(12, 3);
        let mut dl = DictionaryLearning::new(8, 2.0); // 8 atoms for 3 features
        dl.fit(&data, 15).expect("fit overcomplete");
        let codes = dl.transform(&data);
        assert_eq!(codes.len(), 12);
        assert_eq!(codes[0].len(), 8);
    }
}
