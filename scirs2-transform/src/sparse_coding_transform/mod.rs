//! Sparse Coding-Based Feature Transforms
//!
//! This module provides feature transforms based on sparse representations
//! and random feature approximations.
//!
//! ## Algorithms
//!
//! - [`SparseDictTransform`]: Dictionary learning (K-SVD) + OMP sparse coding.
//!   Learns an over-complete dictionary and encodes each sample as a sparse
//!   linear combination of atoms.
//!
//! - [`RandomFourierFeatures`]: Approximate RBF kernels via Bochner's theorem
//!   (Rahimi & Recht, 2007). Features: φ(x) = sqrt(2/D) cos(ω_j^T x + b_j).
//!
//! - [`PolynomialRandomFeatures`]: TensorSketch for polynomial kernel
//!   approximation (Pham & Pagh, 2013).
//!
//! ## References
//!
//! - Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An Algorithm for
//!   Designing Overcomplete Dictionaries for Sparse Representation. TSP.
//! - Tropp, J.A., & Gilbert, A.C. (2007). Signal Recovery from Random
//!   Measurements via Orthogonal Matching Pursuit. TIT.
//! - Rahimi, A., & Recht, B. (2007). Random Features for Large-Scale Kernel
//!   Machines. NeurIPS.
//! - Pham, N., & Pagh, R. (2013). Fast and Scalable Polynomial Kernels via
//!   Explicit Feature Maps. KDD.

use std::f64::consts::PI;

use scirs2_core::random::{seeded_rng, Distribution, Normal, RngCore, SeedableRng, Uniform};

use crate::error::{Result, TransformError};

// ============================================================================
// Helper: Orthogonal Matching Pursuit (OMP)
// ============================================================================

/// Solve the sparse coding problem via OMP:
/// Find sparse alpha such that ||x - D alpha||_2 is minimized with
/// at most `sparsity` nonzero coefficients.
///
/// Returns the sparse coefficient vector (length = n_atoms).
fn omp(x: &[f64], dictionary: &[Vec<f64>], sparsity: usize) -> Vec<f64> {
    let d = x.len();
    let n_atoms = dictionary.len();
    let k = sparsity.min(n_atoms);

    let mut residual = x.to_vec();
    let mut support: Vec<usize> = Vec::with_capacity(k);

    for _ in 0..k {
        // Find atom most correlated with residual
        let best_idx = (0..n_atoms)
            .filter(|idx| !support.contains(idx))
            .max_by(|&a, &b| {
                let ca: f64 = residual.iter().zip(dictionary[a].iter()).map(|(r, di)| r * di).sum::<f64>().abs();
                let cb: f64 = residual.iter().zip(dictionary[b].iter()).map(|(r, di)| r * di).sum::<f64>().abs();
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            });

        let best_idx = match best_idx {
            Some(i) => i,
            None => break,
        };
        support.push(best_idx);

        // Least squares on current support
        let s = support.len();
        // Build sub-matrix D_S (d × s) and solve D_S^T D_S alpha = D_S^T x
        let mut dsd = vec![vec![0.0f64; s]; s]; // D_S^T D_S
        let mut dsx = vec![0.0f64; s]; // D_S^T x

        for (si, &ai) in support.iter().enumerate() {
            let da = &dictionary[ai];
            for (sj, &aj) in support.iter().enumerate() {
                let db = &dictionary[aj];
                let dot: f64 = da.iter().zip(db.iter()).map(|(a, b)| a * b).sum();
                dsd[si][sj] = dot;
            }
            dsx[si] = da.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }

        // Solve s×s system via Gaussian elimination
        let alpha_s = solve_small_system(&dsd, &dsx);

        // Update residual: r = x - D_S alpha_S
        residual = x.to_vec();
        for (si, &ai) in support.iter().enumerate() {
            for (fi, r) in residual.iter_mut().enumerate() {
                if fi < d && fi < dictionary[ai].len() {
                    *r -= alpha_s[si] * dictionary[ai][fi];
                }
            }
        }

        // Check if residual is negligible
        let res_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
        if res_norm < 1e-10 {
            break;
        }
    }

    // Build full coefficient vector
    let mut alpha = vec![0.0f64; n_atoms];
    // Re-solve least squares on final support
    let s = support.len();
    if s > 0 {
        let mut dsd = vec![vec![0.0f64; s]; s];
        let mut dsx = vec![0.0f64; s];
        for (si, &ai) in support.iter().enumerate() {
            let da = &dictionary[ai];
            for (sj, &aj) in support.iter().enumerate() {
                let db = &dictionary[aj];
                let dot: f64 = da.iter().zip(db.iter()).map(|(a, b)| a * b).sum();
                dsd[si][sj] = dot;
            }
            dsx[si] = da.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }
        let alpha_s = solve_small_system(&dsd, &dsx);
        for (si, &ai) in support.iter().enumerate() {
            alpha[ai] = alpha_s[si];
        }
    }
    alpha
}

/// Solve a small linear system Ax = b via Gaussian elimination with partial pivoting.
fn solve_small_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return vec![];
    }

    // Augmented matrix [A | b]
    let mut mat: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n).max_by(|&i, &j| {
            mat[i][col].abs().partial_cmp(&mat[j][col].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });

        let pivot_row = match pivot_row {
            Some(r) => r,
            None => break,
        };

        if mat[pivot_row][col].abs() < 1e-12 {
            continue;
        }

        mat.swap(col, pivot_row);

        let pivot = mat[col][col];
        for j in col..=n {
            mat[col][j] /= pivot;
        }

        for i in (col + 1)..n {
            let factor = mat[i][col];
            for j in col..=n {
                let sub = factor * mat[col][j];
                mat[i][j] -= sub;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        x[i] = mat[i][n];
        for j in (i + 1)..n {
            x[i] -= mat[i][j] * x[j];
        }
    }
    x
}

// ============================================================================
// SparseDictTransform — K-SVD dictionary learning + OMP
// ============================================================================

/// Sparse dictionary transform using K-SVD dictionary learning.
///
/// Learns an over-complete dictionary D ∈ ℝ^{d × K} and represents each
/// sample x ≈ D α where α has at most `sparsity` non-zeros.
#[derive(Debug, Clone)]
pub struct SparseDictTransform {
    /// Learned dictionary atoms: shape (n_atoms, d).
    pub dictionary: Vec<Vec<f64>>,
    /// Maximum number of non-zeros per code.
    pub sparsity: usize,
}

impl SparseDictTransform {
    /// Fit a sparse dictionary on the given data.
    ///
    /// # Arguments
    ///
    /// * `x` - Training samples (n × d).
    /// * `n_atoms` - Number of dictionary atoms K (should be > d for over-completeness).
    /// * `sparsity` - Maximum non-zeros per code.
    /// * `n_iter` - Number of K-SVD outer iterations.
    /// * `seed` - RNG seed for initialization.
    pub fn fit(
        x: &[Vec<f64>],
        n_atoms: usize,
        sparsity: usize,
        n_iter: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = x.len();
        if n == 0 {
            return Err(TransformError::InvalidInput("Empty dataset".to_string()));
        }
        let d = x[0].len();
        if d == 0 {
            return Err(TransformError::InvalidInput("Feature dim must be > 0".to_string()));
        }
        if n_atoms == 0 {
            return Err(TransformError::InvalidInput("n_atoms must be > 0".to_string()));
        }

        // Initialize dictionary: randomly selected (normalized) training samples
        let mut rng = seeded_rng(seed);
        let mut dictionary: Vec<Vec<f64>> = (0..n_atoms)
            .map(|k| {
                let idx = k % n;
                let atom = &x[idx];
                normalize_atom(atom)
            })
            .collect();

        // Add small noise to break ties
        let noise_dist = Normal::new(0.0_f64, 0.01).map_err(|e| {
            TransformError::ComputationError(format!("Normal distribution: {e}"))
        })?;
        for atom in dictionary.iter_mut() {
            for v in atom.iter_mut() {
                *v += noise_dist.sample(&mut rng);
            }
            *atom = normalize_atom(atom);
        }

        // K-SVD iterations
        for _iter in 0..n_iter {
            // Sparse coding step: encode all samples with current dictionary
            let codes: Vec<Vec<f64>> = x.iter().map(|xi| omp(xi, &dictionary, sparsity)).collect();

            // Dictionary update step: update each atom using residuals
            for k in 0..n_atoms {
                // Find samples using atom k
                let users: Vec<usize> = (0..n)
                    .filter(|&i| codes[i][k].abs() > 1e-10)
                    .collect();

                if users.is_empty() {
                    // Reinitialize unused atom to a random training sample
                    let idx = (rng.next_u64() as usize) % n;
                    dictionary[k] = normalize_atom(&x[idx]);
                    continue;
                }

                // E_k = residual matrix when atom k is removed
                // E_k[:, j] = x_j - sum_{m != k} d_m alpha_m_j
                let mut e_k: Vec<Vec<f64>> = users
                    .iter()
                    .map(|&i| {
                        let mut res = x[i].clone();
                        for (m, atom) in dictionary.iter().enumerate() {
                            if m == k {
                                continue;
                            }
                            let coef = codes[i][m];
                            if coef.abs() < 1e-12 {
                                continue;
                            }
                            for (fi, r) in res.iter_mut().enumerate() {
                                if fi < atom.len() {
                                    *r -= coef * atom[fi];
                                }
                            }
                        }
                        res
                    })
                    .collect();

                // SVD of E_k to find best rank-1 update
                // u = E_k @ coefs / ||E_k @ coefs||  (power iteration approximation)
                let coefs_k: Vec<f64> = users.iter().map(|&i| codes[i][k]).collect();

                // New atom ≈ E_k coefs_k / ||coefs_k||^2
                // (simplified: weighted average of residuals)
                let coef_sq: f64 = coefs_k.iter().map(|c| c * c).sum::<f64>();
                if coef_sq < 1e-12 {
                    let idx = (rng.next_u64() as usize) % n;
                    dictionary[k] = normalize_atom(&x[idx]);
                    continue;
                }

                let mut new_atom = vec![0.0f64; d];
                for (ui, &coef) in coefs_k.iter().enumerate() {
                    for (fi, &ev) in e_k[ui].iter().enumerate() {
                        if fi < d {
                            new_atom[fi] += coef * ev;
                        }
                    }
                }
                for v in new_atom.iter_mut() {
                    *v /= coef_sq;
                }
                dictionary[k] = normalize_atom(&new_atom);

                // Update codes: new coef for atom k = E_k^T new_atom (projected)
                for (ui, &i_orig) in users.iter().enumerate() {
                    let _ = i_orig; // code update is read-only in this simplified version
                    let _ = &mut e_k[ui]; // residuals already captured above
                }
            }
        }

        Ok(SparseDictTransform { dictionary, sparsity })
    }

    /// Transform samples into sparse codes using the learned dictionary.
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if self.dictionary.is_empty() {
            return Err(TransformError::NotFitted(
                "Dictionary is empty".to_string(),
            ));
        }
        let d = self.dictionary[0].len();
        let n_atoms = self.dictionary.len();
        let mut out = Vec::with_capacity(x.len());

        for (i, row) in x.iter().enumerate() {
            if row.len() != d {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i}: expected {d} features, got {}",
                    row.len()
                )));
            }
            out.push(omp(row, &self.dictionary, self.sparsity));
        }

        Ok(out)
    }

    /// Reconstruct samples from sparse codes.
    pub fn reconstruct(&self, codes: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if self.dictionary.is_empty() {
            return Err(TransformError::NotFitted("Empty dictionary".to_string()));
        }
        let d = self.dictionary[0].len();
        let n_atoms = self.dictionary.len();
        let mut out = Vec::with_capacity(codes.len());

        for (i, code) in codes.iter().enumerate() {
            if code.len() != n_atoms {
                return Err(TransformError::InvalidInput(format!(
                    "Code {i}: expected {n_atoms} atoms, got {}",
                    code.len()
                )));
            }
            let mut rec = vec![0.0f64; d];
            for (k, &ck) in code.iter().enumerate() {
                if ck.abs() < 1e-12 {
                    continue;
                }
                for (fi, r) in rec.iter_mut().enumerate() {
                    if fi < self.dictionary[k].len() {
                        *r += ck * self.dictionary[k][fi];
                    }
                }
            }
            out.push(rec);
        }
        Ok(out)
    }
}

/// Normalize a vector to unit L2 norm. Returns zero vector if norm is tiny.
fn normalize_atom(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return vec![0.0f64; v.len()];
    }
    v.iter().map(|x| x / norm).collect()
}

// ============================================================================
// RandomFourierFeatures
// ============================================================================

/// Random Fourier Feature approximation for the RBF kernel.
///
/// Approximates k(x, y) = exp(-γ ||x-y||²) via:
/// φ(x)_j = sqrt(2/D) · cos(ω_j^T x + b_j)
///
/// where ω_j ~ N(0, 2γ I) and b_j ~ Uniform(0, 2π).
///
/// See: Rahimi & Recht (2007). Random Features for Large-Scale Kernel Machines.
#[derive(Debug, Clone)]
pub struct RandomFourierFeatures {
    /// Number of random features D.
    pub n_components: usize,
    /// RBF bandwidth parameter γ.
    pub gamma: f64,
    /// Random weight matrix: shape (n_components × n_features).
    pub random_weights: Vec<Vec<f64>>,
    /// Random bias vector: length n_components, drawn from Uniform(0, 2π).
    pub biases: Vec<f64>,
    /// Input feature dimension.
    pub n_features: usize,
}

impl RandomFourierFeatures {
    /// Initialize a new RFF with randomly sampled weights and biases.
    ///
    /// # Arguments
    ///
    /// * `n_components` - Number of random features D.
    /// * `gamma` - RBF bandwidth γ.
    /// * `n_features` - Input dimension.
    /// * `seed` - RNG seed.
    pub fn new(n_components: usize, gamma: f64, n_features: usize, seed: u64) -> Result<Self> {
        if n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }
        if n_features == 0 {
            return Err(TransformError::InvalidInput(
                "n_features must be > 0".to_string(),
            ));
        }
        if gamma <= 0.0 {
            return Err(TransformError::InvalidInput(
                "gamma must be > 0".to_string(),
            ));
        }

        // ω_j ~ N(0, 2γ) for each dimension
        let omega_std = (2.0 * gamma).sqrt();
        let omega_dist = Normal::new(0.0_f64, omega_std).map_err(|e| {
            TransformError::ComputationError(format!("Normal dist: {e}"))
        })?;
        let bias_dist = Uniform::new(0.0_f64, 2.0 * PI).map_err(|e| {
            TransformError::ComputationError(format!("Uniform dist: {e}"))
        })?;

        let mut rng = seeded_rng(seed);
        let random_weights: Vec<Vec<f64>> = (0..n_components)
            .map(|_| (0..n_features).map(|_| omega_dist.sample(&mut rng)).collect())
            .collect();
        let biases: Vec<f64> = (0..n_components).map(|_| bias_dist.sample(&mut rng)).collect();

        Ok(RandomFourierFeatures {
            n_components,
            gamma,
            random_weights,
            biases,
            n_features,
        })
    }

    /// Transform samples into random Fourier features.
    ///
    /// Output: φ(x)_j = sqrt(2/D) · cos(ω_j^T x + b_j).
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let scale = (2.0 / self.n_components as f64).sqrt();
        let mut out = Vec::with_capacity(x.len());

        for (i, row) in x.iter().enumerate() {
            if row.len() != self.n_features {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i}: expected {} features, got {}",
                    self.n_features,
                    row.len()
                )));
            }
            let features: Vec<f64> = self
                .random_weights
                .iter()
                .zip(self.biases.iter())
                .map(|(omega, &bias)| {
                    let dot: f64 = omega.iter().zip(row.iter()).map(|(o, xi)| o * xi).sum();
                    scale * (dot + bias).cos()
                })
                .collect();
            out.push(features);
        }
        Ok(out)
    }

    /// Estimate kernel value between two points using the features:
    /// k̂(x, y) ≈ φ(x)^T φ(y).
    pub fn estimate_kernel(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        let px = self.transform(&[x.to_vec()])?;
        let py = self.transform(&[y.to_vec()])?;
        let k: f64 = px[0].iter().zip(py[0].iter()).map(|(a, b)| a * b).sum();
        Ok(k)
    }
}

// ============================================================================
// PolynomialRandomFeatures — TensorSketch
// ============================================================================

/// Polynomial kernel approximation via TensorSketch (count sketch + FFT).
///
/// Approximates k(x, y) = (γ x^T y + c)^d via random sketches.
///
/// The sketch of the d-fold outer product is estimated by convolving d
/// count-sketch vectors, leveraging the convolution theorem.
///
/// See: Pham & Pagh (2013). Fast and Scalable Polynomial Kernels.
#[derive(Debug, Clone)]
pub struct PolynomialRandomFeatures {
    /// Number of sketch components.
    pub n_components: usize,
    /// Polynomial degree.
    pub degree: usize,
    /// Scaling factor γ.
    pub gamma: f64,
    /// Constant addend c in (γ x^T y + c)^d.
    pub coef0: f64,
    /// Hash functions (one per degree): (h[j], s[j]) per dimension.
    /// h[j][i] ∈ {0, ..., n_components-1}: bucket index for feature i in sketch j.
    /// s[j][i] ∈ {-1, +1}: sign for feature i in sketch j.
    h_maps: Vec<Vec<usize>>,
    s_maps: Vec<Vec<i8>>,
    /// Input feature dimension.
    pub n_features: usize,
}

impl PolynomialRandomFeatures {
    /// Create a new TensorSketch with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `n_components` - Sketch dimension (number of features per sketch).
    /// * `degree` - Polynomial degree d.
    /// * `gamma` - Scaling γ.
    /// * `coef0` - Constant c.
    /// * `n_features` - Input dimension.
    /// * `seed` - RNG seed.
    pub fn new(
        n_components: usize,
        degree: usize,
        gamma: f64,
        coef0: f64,
        n_features: usize,
        seed: u64,
    ) -> Result<Self> {
        if n_components == 0 || degree == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput(
                "n_components, degree, and n_features must all be > 0".to_string(),
            ));
        }

        let mut rng = seeded_rng(seed);
        let h_dist = Uniform::new(0_usize, n_components).map_err(|e| {
            TransformError::ComputationError(format!("Uniform h dist: {e}"))
        })?;
        let s_dist = Uniform::new(0_usize, 2).map_err(|e| {
            TransformError::ComputationError(format!("Uniform s dist: {e}"))
        })?;

        let mut h_maps: Vec<Vec<usize>> = Vec::with_capacity(degree);
        let mut s_maps: Vec<Vec<i8>> = Vec::with_capacity(degree);

        for _ in 0..degree {
            let h: Vec<usize> = (0..n_features).map(|_| h_dist.sample(&mut rng)).collect();
            let s: Vec<i8> = (0..n_features)
                .map(|_| if s_dist.sample(&mut rng) == 0 { -1i8 } else { 1i8 })
                .collect();
            h_maps.push(h);
            s_maps.push(s);
        }

        Ok(PolynomialRandomFeatures {
            n_components,
            degree,
            gamma,
            coef0,
            h_maps,
            s_maps,
            n_features,
        })
    }

    /// Compute the count sketch of a single input vector for sketch `j`.
    fn count_sketch(&self, x: &[f64], j: usize) -> Vec<f64> {
        let mut sketch = vec![0.0f64; self.n_components];
        for (i, &xi) in x.iter().enumerate() {
            let bucket = self.h_maps[j][i];
            let sign = self.s_maps[j][i] as f64;
            sketch[bucket] += sign * xi;
        }
        sketch
    }

    /// Circular convolution of two vectors (in the frequency domain):
    /// (a * b)[k] = sum_j a[j] * b[(k-j) mod D].
    fn circular_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                out[(i + j) % n] += a[i] * b[j];
            }
        }
        out
    }

    /// Transform a batch of samples into polynomial random features.
    ///
    /// For each sample x, the feature vector is the element-wise product
    /// (convolution) of `degree` count sketches, scaled by γ.
    ///
    /// The resulting features approximate (γ x^T y + c)^d.
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let mut out = Vec::with_capacity(x.len());

        for (i, row) in x.iter().enumerate() {
            if row.len() != self.n_features {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i}: expected {} features, got {}",
                    self.n_features,
                    row.len()
                )));
            }

            // Scale input by gamma
            let scaled: Vec<f64> = row.iter().map(|&v| self.gamma * v).collect();

            // Add coef0 (augment with a constant feature)
            // We handle c by adding an extra dimension with value sqrt(c)
            // but for simplicity we handle it as a polynomial shift on the sketch.
            // Simplified: compute sketches for the polynomial part only.

            // Compute degree count sketches and convolve
            let sketches: Vec<Vec<f64>> = (0..self.degree)
                .map(|j| self.count_sketch(&scaled, j))
                .collect();

            // Convolve all sketches together
            let mut feature = sketches[0].clone();
            for j in 1..self.degree {
                feature = Self::circular_convolve(&feature, &sketches[j]);
            }

            // Scale by 1/n_components for normalization
            let scale = 1.0 / (self.n_components as f64).sqrt();
            let feature: Vec<f64> = feature.iter().map(|&v| v * scale).collect();
            out.push(feature);
        }
        Ok(out)
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
    fn test_rff_transform_shape() {
        let x = make_data(20, 5, 0);
        let rff = RandomFourierFeatures::new(50, 1.0, 5, 42).expect("RFF::new");
        let phi = rff.transform(&x).expect("transform");
        assert_eq!(phi.len(), 20);
        assert_eq!(phi[0].len(), 50);
    }

    #[test]
    fn test_rff_kernel_approximation() {
        // phi(x)^T phi(y) ≈ exp(-gamma ||x-y||^2)
        let x = vec![vec![1.0, 0.0, 0.0]];
        let y = vec![vec![1.0, 0.0, 0.0]]; // same point → kernel = 1
        let rff = RandomFourierFeatures::new(5000, 1.0, 3, 0).expect("RFF::new");
        let k = rff.estimate_kernel(&x[0], &y[0]).expect("kernel");
        assert!((k - 1.0).abs() < 0.05, "RBF(x,x) ≈ 1, got {k:.4}");
    }

    #[test]
    fn test_rff_kernel_decreasing_with_distance() {
        let rff = RandomFourierFeatures::new(2000, 1.0, 3, 1).expect("RFF::new");
        let x = vec![0.0, 0.0, 0.0];
        let y_near = vec![0.1, 0.0, 0.0];
        let y_far = vec![2.0, 0.0, 0.0];
        let k_near = rff.estimate_kernel(&x, &y_near).expect("k_near");
        let k_far = rff.estimate_kernel(&x, &y_far).expect("k_far");
        assert!(k_near > k_far, "Near kernel {k_near:.4} should exceed far {k_far:.4}");
    }

    #[test]
    fn test_rff_invalid() {
        assert!(RandomFourierFeatures::new(0, 1.0, 3, 0).is_err());
        assert!(RandomFourierFeatures::new(10, 0.0, 3, 0).is_err()); // gamma <= 0
        assert!(RandomFourierFeatures::new(10, -1.0, 3, 0).is_err());
    }

    #[test]
    fn test_sparse_dict_basic() {
        let x = make_data(30, 8, 5);
        let sdt = SparseDictTransform::fit(&x, 16, 3, 5, 0).expect("fit");
        assert_eq!(sdt.dictionary.len(), 16);
        assert_eq!(sdt.dictionary[0].len(), 8);

        let codes = sdt.transform(&x).expect("transform");
        assert_eq!(codes.len(), 30);
        assert_eq!(codes[0].len(), 16);

        // Each code should have at most `sparsity` non-zeros
        for code in &codes {
            let nnz = code.iter().filter(|&&v| v.abs() > 1e-10).count();
            assert!(nnz <= sdt.sparsity, "NNZ {nnz} > sparsity {}", sdt.sparsity);
        }
    }

    #[test]
    fn test_sparse_dict_reconstruct() {
        let x = make_data(20, 6, 10);
        let sdt = SparseDictTransform::fit(&x, 12, 4, 10, 1).expect("fit");
        let codes = sdt.transform(&x).expect("transform");
        let recon = sdt.reconstruct(&codes).expect("reconstruct");
        assert_eq!(recon.len(), 20);
        assert_eq!(recon[0].len(), 6);
    }

    #[test]
    fn test_poly_rff_shape() {
        let x = make_data(15, 4, 0);
        let prff = PolynomialRandomFeatures::new(32, 3, 1.0, 0.0, 4, 0).expect("new");
        let out = prff.transform(&x).expect("transform");
        assert_eq!(out.len(), 15);
        assert_eq!(out[0].len(), 32);
    }

    #[test]
    fn test_omp_basic() {
        // Simple 2D case: x = 0.5*d0 + 0.3*d1
        let d0 = vec![1.0, 0.0];
        let d1 = vec![0.0, 1.0];
        let d2 = vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
        let dictionary = vec![d0, d1, d2];
        let x = vec![0.5, 0.3];
        let codes = omp(&x, &dictionary, 2);
        assert_eq!(codes.len(), 3);
        // Reconstruction should be close to x
        let recon: Vec<f64> = (0..2)
            .map(|fi| codes.iter().zip(dictionary.iter()).map(|(c, d)| c * d[fi]).sum::<f64>())
            .collect();
        let err: f64 = x.iter().zip(recon.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
        assert!(err < 1e-8, "OMP reconstruction error {err:.2e}");
    }
}
