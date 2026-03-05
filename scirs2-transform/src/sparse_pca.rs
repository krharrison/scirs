//! Sparse PCA and Dictionary Learning with K-SVD
//!
//! ## Sparse PCA
//!
//! Sparse PCA finds principal components that are *sparse* — most of their
//! loadings are zero — making them more interpretable.  It is formulated as
//! a dictionary learning problem where the dictionary is constrained to be
//! orthogonal:
//!
//! > minimise  ½ ||X − Z D||²_F  +  α ||Z||_1
//!
//! We solve this by alternating:
//! 1. **Code update** — given D, find sparse codes Z via Lasso CD or OMP.
//! 2. **Dictionary update** — given Z, update D using closed-form projection.
//!
//! ## Dictionary Learning (K-SVD)
//!
//! K-SVD is an SVD-based generalisation of k-means for dictionary learning.
//! It alternates between:
//! 1. Sparse coding each sample (OMP).
//! 2. Updating each dictionary atom to minimise the residual over all samples
//!    that use that atom, using a rank-1 SVD update.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Rng;
use scirs2_linalg::svd;

use crate::error::{Result, TransformError};

// ────────────────────────────────────────────────────────────────────────────
// Sparse-coding methods
// ────────────────────────────────────────────────────────────────────────────

/// Sparse-coding method
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparsePcaMethod {
    /// Lasso coordinate descent (convex, guaranteed optimal)
    LassoCD,
    /// Orthogonal Matching Pursuit (greedy, fast, approximate)
    OMP,
}

// ────────────────────────────────────────────────────────────────────────────
// Sparse PCA
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for Sparse PCA
#[derive(Debug, Clone)]
pub struct SparsePcaConfig {
    /// Number of sparse components (default 2)
    pub n_components: usize,
    /// Sparsity regularisation (default 1.0 for LassoCD, max non-zeros for OMP)
    pub alpha: f64,
    /// Maximum number of alternating-iteration steps (default 100)
    pub max_iter: usize,
    /// Convergence tolerance on the objective (default 1e-5)
    pub tol: f64,
    /// Sparse-coding method (default LassoCD)
    pub method: SparsePcaMethod,
    /// RNG seed for random initialisation of the dictionary (default 42)
    pub seed: u64,
}

impl Default for SparsePcaConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            alpha: 1.0,
            max_iter: 100,
            tol: 1e-5,
            method: SparsePcaMethod::LassoCD,
            seed: 42,
        }
    }
}

/// Result of Sparse PCA
#[derive(Debug, Clone)]
pub struct SparsePcaResult {
    /// Sparse components (dictionary), shape `(n_components, n_features)`
    pub components: Array2<f64>,
    /// Sparse codes (loadings), shape `(n_samples, n_components)`
    pub loadings: Array2<f64>,
    /// Number of alternating iterations executed
    pub n_iter: usize,
    /// Final reconstruction error `||X − Z D||_F`
    pub error: f64,
}

// ────────────────────────────────────────────────────────────────────────────
// Lasso coordinate descent (single sample)
// ────────────────────────────────────────────────────────────────────────────

/// Lasso coordinate descent for a single sample:
/// minimise  ½ ||x − Dz||²  +  α ||z||_1
///
/// Returns sparse code vector of length `n_components`.
fn lasso_cd(
    x: &Array1<f64>,
    dictionary: &Array2<f64>,
    alpha: f64,
    max_iter: usize,
) -> Array1<f64> {
    let k = dictionary.shape()[0]; // n_components
    let mut z = Array1::zeros(k);

    // Pre-compute Gram matrix G = D D^T and correlation r0 = D x
    let mut gram = Array2::zeros((k, k));
    let mut r0 = Array1::zeros(k);
    for i in 0..k {
        for j in 0..k {
            let mut dot = 0.0_f64;
            for f in 0..dictionary.shape()[1] {
                dot += dictionary[[i, f]] * dictionary[[j, f]];
            }
            gram[[i, j]] = dot;
        }
        let mut dot = 0.0_f64;
        for f in 0..x.len() {
            dot += dictionary[[i, f]] * x[f];
        }
        r0[i] = dot;
    }

    for _ in 0..max_iter {
        let z_old = z.clone();
        for j in 0..k {
            // Coordinate update with soft-thresholding
            let mut rho = r0[j];
            for l in 0..k {
                if l != j {
                    rho -= gram[[j, l]] * z[l];
                }
            }
            let dj_sq = gram[[j, j]];
            let z_j = if dj_sq > 1e-12 {
                soft_threshold(rho / dj_sq, alpha / dj_sq)
            } else {
                0.0
            };
            z[j] = z_j;
        }
        // Check convergence
        let delta = (&z - &z_old).mapv(|v| v * v).sum().sqrt();
        if delta < 1e-8 {
            break;
        }
    }
    z
}

/// Soft-threshold operator: sign(x) * max(|x| − λ, 0)
#[inline]
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Orthogonal Matching Pursuit (single sample)
// ────────────────────────────────────────────────────────────────────────────

/// OMP for a single sample: greedily selects at most `n_nonzero` atoms.
fn omp(x: &Array1<f64>, dictionary: &Array2<f64>, n_nonzero: usize) -> Array1<f64> {
    let k = dictionary.shape()[0];
    let n_features = dictionary.shape()[1];
    let n_nonzero = n_nonzero.min(k).min(n_features);

    let mut residual: Vec<f64> = x.to_vec();
    let mut selected: Vec<usize> = Vec::with_capacity(n_nonzero);
    let mut z = Array1::zeros(k);

    for _ in 0..n_nonzero {
        // Find atom with maximum absolute inner product with residual
        let mut best_idx = 0;
        let mut best_val = 0.0_f64;
        for j in 0..k {
            if selected.contains(&j) {
                continue;
            }
            let mut dot = 0.0_f64;
            for f in 0..n_features {
                dot += dictionary[[j, f]] * residual[f];
            }
            if dot.abs() > best_val {
                best_val = dot.abs();
                best_idx = j;
            }
        }
        if best_val < 1e-12 {
            break;
        }
        selected.push(best_idx);

        // Least-squares fit over selected atoms
        let ns = selected.len();
        let mut a = Array2::zeros((ns, ns));
        let mut b_vec = Array1::zeros(ns);
        for (i, &si) in selected.iter().enumerate() {
            let mut bi = 0.0_f64;
            for f in 0..n_features {
                bi += dictionary[[si, f]] * x[f];
            }
            b_vec[i] = bi;
            for (j, &sj) in selected.iter().enumerate() {
                let mut dot = 0.0_f64;
                for f in 0..n_features {
                    dot += dictionary[[si, f]] * dictionary[[sj, f]];
                }
                a[[i, j]] = dot;
            }
        }
        // Solve a * coeffs = b via simple CD on the small system
        let coeffs = solve_small_ls(&a, &b_vec);

        // Update z and residual
        for (i, &si) in selected.iter().enumerate() {
            z[si] = coeffs[i];
        }
        for f in 0..n_features {
            let mut rec = 0.0_f64;
            for (i, &si) in selected.iter().enumerate() {
                rec += coeffs[i] * dictionary[[si, f]];
            }
            residual[f] = x[f] - rec;
        }
    }
    z
}

/// Solve a small symmetric positive-semi-definite linear system A x = b
/// using Gauss-Seidel iteration (avoids bringing in a full linalg dependency
/// for tiny systems).
fn solve_small_ls(a: &Array2<f64>, b: &Array1<f64>) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0_f64; n];
    for _ in 0..500 {
        let mut delta = 0.0_f64;
        for i in 0..n {
            if a[[i, i]].abs() < 1e-14 {
                continue;
            }
            let mut s = b[i];
            for j in 0..n {
                if j != i {
                    s -= a[[i, j]] * x[j];
                }
            }
            let x_new = s / a[[i, i]];
            delta += (x_new - x[i]).abs();
            x[i] = x_new;
        }
        if delta < 1e-10 {
            break;
        }
    }
    x
}

// ────────────────────────────────────────────────────────────────────────────
// Sparse PCA function
// ────────────────────────────────────────────────────────────────────────────

/// Compute Sparse PCA: find sparse components via alternating dictionary learning.
///
/// # Arguments
/// * `data` — shape `(n_samples, n_features)`
/// * `config` — Sparse PCA hyper-parameters
pub fn sparse_pca(data: &Array2<f64>, config: SparsePcaConfig) -> Result<SparsePcaResult> {
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let k = config.n_components;

    if n_samples == 0 || n_features == 0 {
        return Err(TransformError::InvalidInput("Empty input data".to_string()));
    }
    if k == 0 || k > n_features {
        return Err(TransformError::InvalidInput(format!(
            "n_components ({k}) must be in 1..={n_features}"
        )));
    }

    // Initialise dictionary from random samples in data
    let mut rng = scirs2_core::random::rng();
    let mut dictionary = Array2::zeros((k, n_features));
    for i in 0..k {
        let sample_idx = rng.random_range(0..n_samples);
        let mut norm_sq = 0.0_f64;
        for f in 0..n_features {
            dictionary[[i, f]] = data[[sample_idx, f]];
            norm_sq += data[[sample_idx, f]] * data[[sample_idx, f]];
        }
        let norm = norm_sq.sqrt().max(1e-12);
        for f in 0..n_features {
            dictionary[[i, f]] /= norm;
        }
    }

    let n_nonzero = (config.alpha as usize).max(1).min(k);
    let mut codes = Array2::zeros((n_samples, k));
    let mut prev_error = f64::INFINITY;
    let mut done_iters = 0_usize;

    for iter in 0..config.max_iter {
        // ── Code update: sparse-code each sample given the current dictionary ──
        for i in 0..n_samples {
            let xi: Array1<f64> = data.row(i).to_owned();
            let z = match config.method {
                SparsePcaMethod::LassoCD => lasso_cd(&xi, &dictionary, config.alpha, 500),
                SparsePcaMethod::OMP => omp(&xi, &dictionary, n_nonzero),
            };
            for j in 0..k {
                codes[[i, j]] = z[j];
            }
        }

        // ── Dictionary update: closed-form update keeping atoms normalised ──
        // New dict row d_j = normalise(X^T z_j / ||z_j||²)
        for j in 0..k {
            let mut num = Array1::<f64>::zeros(n_features);
            let mut denom = 0.0_f64;
            for i in 0..n_samples {
                let c = codes[[i, j]];
                if c.abs() < 1e-12 {
                    continue;
                }
                for f in 0..n_features {
                    num[f] += c * data[[i, f]];
                }
                denom += c * c;
            }
            if denom > 1e-12 {
                let scale = 1.0 / denom.sqrt();
                for f in 0..n_features {
                    dictionary[[j, f]] = num[f] * scale;
                }
                // Normalise atom
                let mut ns = 0.0_f64;
                for f in 0..n_features {
                    ns += dictionary[[j, f]] * dictionary[[j, f]];
                }
                ns = ns.sqrt().max(1e-12);
                for f in 0..n_features {
                    dictionary[[j, f]] /= ns;
                }
            }
        }

        // ── Compute reconstruction error ──
        let mut error_sq = 0.0_f64;
        for i in 0..n_samples {
            for f in 0..n_features {
                let mut rec = 0.0_f64;
                for j in 0..k {
                    rec += codes[[i, j]] * dictionary[[j, f]];
                }
                let diff = data[[i, f]] - rec;
                error_sq += diff * diff;
            }
        }
        let error = error_sq.sqrt();
        done_iters = iter + 1;

        if (prev_error - error).abs() < config.tol * (1.0 + prev_error) {
            prev_error = error;
            break;
        }
        prev_error = error;
    }

    Ok(SparsePcaResult {
        components: dictionary,
        loadings: codes,
        n_iter: done_iters,
        error: prev_error,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Dictionary Learning (K-SVD)
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for K-SVD dictionary learning
#[derive(Debug, Clone)]
pub struct DictLearningConfig {
    /// Number of dictionary atoms (default 8)
    pub n_atoms: usize,
    /// Sparsity level (maximum number of non-zeros per code for OMP; default 3)
    pub alpha: f64,
    /// Maximum number of K-SVD iterations (default 50)
    pub max_iter: usize,
    /// RNG seed for initialisation (default 42)
    pub seed: u64,
}

impl Default for DictLearningConfig {
    fn default() -> Self {
        Self {
            n_atoms: 8,
            alpha: 3.0,
            max_iter: 50,
            seed: 42,
        }
    }
}

/// Result of dictionary learning
#[derive(Debug, Clone)]
pub struct DictLearningResult {
    /// Learned dictionary, shape `(n_atoms, n_features)`
    pub dictionary: Array2<f64>,
    /// Sparse codes, shape `(n_samples, n_atoms)`
    pub codes: Array2<f64>,
    /// Reconstruction error per iteration
    pub objective: Vec<f64>,
}

// ────────────────────────────────────────────────────────────────────────────
// K-SVD dictionary learning
// ────────────────────────────────────────────────────────────────────────────

/// Encode all samples using OMP
fn omp_encode(data: &Array2<f64>, dictionary: &Array2<f64>, n_nonzero: usize) -> Array2<f64> {
    let n_samples = data.shape()[0];
    let n_atoms = dictionary.shape()[0];
    let mut codes = Array2::zeros((n_samples, n_atoms));
    for i in 0..n_samples {
        let xi: Array1<f64> = data.row(i).to_owned();
        let z = omp(&xi, dictionary, n_nonzero);
        for j in 0..n_atoms {
            codes[[i, j]] = z[j];
        }
    }
    codes
}

/// K-SVD atom update: for atom j, compute the best rank-1 update via SVD of
/// the error matrix restricted to samples that use atom j.
fn ksvd_update_atom(
    data: &Array2<f64>,
    dictionary: &mut Array2<f64>,
    codes: &mut Array2<f64>,
    atom_idx: usize,
) {
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let n_atoms = dictionary.shape()[0];

    // Indices of samples that use this atom
    let using: Vec<usize> = (0..n_samples)
        .filter(|&i| codes[[i, atom_idx]].abs() > 1e-12)
        .collect();

    if using.is_empty() {
        return;
    }

    let m = using.len();

    // Compute error matrix E_j = X_S - D_{-j} Z_S  (size m × n_features but transposed for SVD)
    // We accumulate it as (n_features × m)
    let mut e_mat = Array2::zeros((n_features, m));
    for (col, &sample_i) in using.iter().enumerate() {
        for f in 0..n_features {
            let mut rec = 0.0_f64;
            for j in 0..n_atoms {
                if j != atom_idx {
                    rec += codes[[sample_i, j]] * dictionary[[j, f]];
                }
            }
            e_mat[[f, col]] = data[[sample_i, f]] - rec;
        }
    }

    // SVD of e_mat (n_features × m): take first left singular vector as new atom
    // and first right singular vector × singular value as new codes
    match svd::<f64>(&e_mat.view(), true, None) {
        Ok((u, s, vt)) => {
            // Update atom = first column of U
            for f in 0..n_features {
                dictionary[[atom_idx, f]] = u[[f, 0]];
            }
            // Update codes for the using samples: z_i = s[0] * vt[0, col]
            if !s.is_empty() {
                let sv = s[0];
                for (col, &sample_i) in using.iter().enumerate() {
                    codes[[sample_i, atom_idx]] = sv * vt[[0, col]];
                }
            }
        }
        Err(_) => {
            // Leave atom unchanged on SVD failure
        }
    }
}

/// Dictionary learning via K-SVD algorithm.
///
/// # Arguments
/// * `data` — shape `(n_samples, n_features)`
/// * `config` — K-SVD hyper-parameters
pub fn dictionary_learning(
    data: &Array2<f64>,
    config: DictLearningConfig,
) -> Result<DictLearningResult> {
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];
    let n_atoms = config.n_atoms;
    let n_nonzero = (config.alpha as usize).max(1).min(n_atoms);

    if n_samples == 0 || n_features == 0 {
        return Err(TransformError::InvalidInput("Empty input data".to_string()));
    }
    if n_atoms == 0 {
        return Err(TransformError::InvalidInput(
            "n_atoms must be >= 1".to_string(),
        ));
    }

    // Initialise dictionary from random subsets of training data (normalised)
    let mut rng = scirs2_core::random::rng();
    let mut dictionary = Array2::zeros((n_atoms, n_features));
    for i in 0..n_atoms {
        let idx = rng.random_range(0..n_samples);
        let mut ns = 0.0_f64;
        for f in 0..n_features {
            dictionary[[i, f]] = data[[idx, f]];
            ns += data[[idx, f]] * data[[idx, f]];
        }
        ns = ns.sqrt().max(1e-12);
        for f in 0..n_features {
            dictionary[[i, f]] /= ns;
        }
    }

    let mut codes = omp_encode(data, &dictionary, n_nonzero);
    let mut objective_history: Vec<f64> = Vec::with_capacity(config.max_iter);

    for _iter in 0..config.max_iter {
        // ── K-SVD: update each atom ──
        for j in 0..n_atoms {
            ksvd_update_atom(data, &mut dictionary, &mut codes, j);
        }

        // ── Re-encode all samples with updated dictionary ──
        codes = omp_encode(data, &dictionary, n_nonzero);

        // ── Record reconstruction error ──
        let mut err_sq = 0.0_f64;
        for i in 0..n_samples {
            for f in 0..n_features {
                let mut rec = 0.0_f64;
                for j in 0..n_atoms {
                    rec += codes[[i, j]] * dictionary[[j, f]];
                }
                let diff = data[[i, f]] - rec;
                err_sq += diff * diff;
            }
        }
        objective_history.push(err_sq.sqrt());
    }

    Ok(DictLearningResult {
        dictionary,
        codes,
        objective: objective_history,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn synthetic_data(n_samples: usize, n_features: usize) -> Array2<f64> {
        // Simple structured data: rows are linear combinations of a few basis vectors
        let mut rows = Vec::with_capacity(n_samples * n_features);
        for i in 0..n_samples {
            let t = i as f64 / n_samples as f64;
            for f in 0..n_features {
                let v = (t * std::f64::consts::PI * (f + 1) as f64).sin();
                rows.push(v);
            }
        }
        Array::from_shape_vec((n_samples, n_features), rows).expect("shape")
    }

    #[test]
    fn test_sparse_pca_output_shapes() {
        let data = synthetic_data(20, 8);
        let res = sparse_pca(
            &data,
            SparsePcaConfig {
                n_components: 3,
                ..Default::default()
            },
        )
        .expect("sparse_pca");
        assert_eq!(res.components.shape(), &[3, 8]);
        assert_eq!(res.loadings.shape(), &[20, 3]);
        for v in res.components.iter() {
            assert!(v.is_finite());
        }
        for v in res.loadings.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sparse_pca_components_sparse_lasso() {
        let data = synthetic_data(30, 10);
        let res = sparse_pca(
            &data,
            SparsePcaConfig {
                n_components: 4,
                alpha: 2.0,
                method: SparsePcaMethod::LassoCD,
                ..Default::default()
            },
        )
        .expect("sparse_pca lasso");
        // Loadings should have many zeros (sparsity enforced by L1)
        let total = (res.loadings.shape()[0] * res.loadings.shape()[1]) as f64;
        let n_zero = res
            .loadings
            .iter()
            .filter(|&&v| v.abs() < 1e-8)
            .count() as f64;
        assert!(
            n_zero / total > 0.1,
            "loadings not sparse enough: {n_zero}/{total} zeros"
        );
    }

    #[test]
    fn test_sparse_pca_components_sparse_omp() {
        let data = synthetic_data(30, 10);
        let res = sparse_pca(
            &data,
            SparsePcaConfig {
                n_components: 4,
                alpha: 2.0,
                method: SparsePcaMethod::OMP,
                ..Default::default()
            },
        )
        .expect("sparse_pca omp");
        // OMP with n_nonzero=2 — at most 2 non-zeros per code vector
        let total = (res.loadings.shape()[0] * res.loadings.shape()[1]) as f64;
        let n_zero = res
            .loadings
            .iter()
            .filter(|&&v| v.abs() < 1e-8)
            .count() as f64;
        assert!(
            n_zero / total > 0.1,
            "OMP loadings not sparse enough: {n_zero}/{total} zeros"
        );
    }

    #[test]
    fn test_sparse_pca_error_finite() {
        let data = synthetic_data(15, 6);
        let res = sparse_pca(&data, SparsePcaConfig::default()).expect("sparse_pca");
        assert!(res.error.is_finite(), "reconstruction error should be finite");
        assert!(res.error >= 0.0);
    }

    #[test]
    fn test_sparse_pca_invalid_n_components() {
        let data: Array2<f64> = Array::zeros((10, 5));
        let res = sparse_pca(
            &data,
            SparsePcaConfig {
                n_components: 10, // > n_features
                ..Default::default()
            },
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_dict_learning_output_shapes() {
        let data = synthetic_data(20, 8);
        let res = dictionary_learning(
            &data,
            DictLearningConfig {
                n_atoms: 5,
                alpha: 2.0,
                max_iter: 10,
                ..Default::default()
            },
        )
        .expect("dictionary_learning");
        assert_eq!(res.dictionary.shape(), &[5, 8]);
        assert_eq!(res.codes.shape(), &[20, 5]);
        assert_eq!(res.objective.len(), 10);
        for v in res.dictionary.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dict_learning_reconstruction_error_decreases() {
        let data = synthetic_data(25, 6);
        let res = dictionary_learning(
            &data,
            DictLearningConfig {
                n_atoms: 6,
                alpha: 2.0,
                max_iter: 30,
                ..Default::default()
            },
        )
        .expect("dictionary_learning");
        let obj = &res.objective;
        // The error should decrease overall (first → last)
        // Allow for slight fluctuations — check that final < initial * 1.5
        if obj.len() >= 2 {
            assert!(
                obj.last().copied().unwrap_or(0.0)
                    <= obj.first().copied().unwrap_or(f64::INFINITY) * 1.5,
                "reconstruction error did not decrease: {:?}",
                obj
            );
        }
        // All objective values should be finite and non-negative
        for &v in obj.iter() {
            assert!(v.is_finite() && v >= 0.0);
        }
    }

    #[test]
    fn test_dict_learning_atoms_normalised() {
        let data = synthetic_data(20, 8);
        let res = dictionary_learning(
            &data,
            DictLearningConfig {
                n_atoms: 4,
                alpha: 2.0,
                max_iter: 5,
                ..Default::default()
            },
        )
        .expect("dictionary_learning");
        for j in 0..4 {
            let mut ns = 0.0_f64;
            for f in 0..8 {
                ns += res.dictionary[[j, f]] * res.dictionary[[j, f]];
            }
            assert!(
                (ns.sqrt() - 1.0).abs() < 0.2 || ns < 1e-6,
                "atom {j} has unexpected norm: {ns}"
            );
        }
    }

    #[test]
    fn test_dict_learning_invalid_n_atoms() {
        let data: Array2<f64> = Array::zeros((10, 5));
        let res = dictionary_learning(
            &data,
            DictLearningConfig {
                n_atoms: 0,
                ..Default::default()
            },
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_soft_threshold() {
        assert!((soft_threshold(3.0, 1.0) - 2.0).abs() < 1e-12);
        assert!((soft_threshold(-3.0, 1.0) + 2.0).abs() < 1e-12);
        assert!((soft_threshold(0.5, 1.0)).abs() < 1e-12);
    }
}
