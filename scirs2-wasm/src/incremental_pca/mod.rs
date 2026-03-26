//! Incremental PCA for streaming/online dimensionality reduction in WASM
//!
//! Implements the incremental SVD update algorithm from Ross et al. (2008)
//! "Incremental Learning for Robust Visual Tracking."
//!
//! Suitable for large datasets that cannot fit in memory at once, or for
//! online learning scenarios in browser environments.

use crate::error::{WasmError, WasmResult};

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for Incremental PCA
#[derive(Debug, Clone)]
pub struct IncrementalPcaConfig {
    /// Number of principal components to retain
    pub n_components: usize,
    /// Default batch size for `fit()`
    pub batch_size: usize,
    /// Whiten the output (divide by singular values)
    pub whiten: bool,
}

impl Default for IncrementalPcaConfig {
    fn default() -> Self {
        IncrementalPcaConfig {
            n_components: 10,
            batch_size: 100,
            whiten: false,
        }
    }
}

// ─── Incremental PCA struct ───────────────────────────────────────────────────

/// Online/streaming Principal Component Analysis.
///
/// Call `partial_fit` repeatedly with mini-batches. After all batches have been
/// processed, use `transform` to project new data.
#[derive(Debug, Clone)]
pub struct IncrementalPca {
    /// Principal components (n_components × n_features)
    pub components: Vec<Vec<f64>>,
    /// Singular values corresponding to each component
    pub singular_values: Vec<f64>,
    /// Running feature mean
    pub mean: Vec<f64>,
    /// Total number of samples seen so far
    pub n_samples_seen: usize,
    /// Number of input features
    pub n_features: usize,
    /// Number of components to retain
    n_components: usize,
    /// Whether to whiten
    whiten: bool,
}

impl IncrementalPca {
    /// Create a new `IncrementalPca` with `n_components` components and known
    /// feature dimensionality.
    pub fn new(n_components: usize, n_features: usize) -> Self {
        IncrementalPca {
            components: Vec::new(),
            singular_values: Vec::new(),
            mean: vec![0.0_f64; n_features],
            n_samples_seen: 0,
            n_features,
            n_components,
            whiten: false,
        }
    }

    /// Set whitening on or off.
    pub fn set_whiten(&mut self, whiten: bool) {
        self.whiten = whiten;
    }

    /// Process one mini-batch of data and update the PCA model.
    ///
    /// `x` is an n_samples × n_features matrix (rows are samples).
    pub fn partial_fit(&mut self, x: &[Vec<f64>]) -> WasmResult<()> {
        if x.is_empty() {
            return Ok(());
        }
        let n_samples = x.len();
        let n_features = x[0].len();

        if self.n_features != n_features {
            return Err(WasmError::InvalidDimensions(format!(
                "Expected {}-feature samples, got {}",
                self.n_features, n_features
            )));
        }

        // ── Update running mean ──────────────────────────────────────────────
        let n_total = self.n_samples_seen + n_samples;
        let mut new_mean = vec![0.0_f64; n_features];
        for f in 0..n_features {
            let batch_mean: f64 = x.iter().map(|row| row[f]).sum::<f64>() / n_samples as f64;
            new_mean[f] = (self.mean[f] * self.n_samples_seen as f64
                + batch_mean * n_samples as f64)
                / n_total as f64;
        }
        let old_mean = std::mem::replace(&mut self.mean, new_mean.clone());
        self.n_samples_seen = n_total;

        // ── Centre batch: X_c = X - new_mean ────────────────────────────────
        let x_c: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(f, &v)| v - new_mean[f])
                    .collect()
            })
            .collect();

        // ── Build augmented matrix ───────────────────────────────────────────
        // If we already have components, prepend  components * diag(singular_values)
        // corrected for the mean shift.
        let mut augmented: Vec<Vec<f64>> = Vec::new();

        if !self.components.is_empty() {
            let k = self.components.len();
            // Mean correction row: sqrt(n_old * n_new / n_total) * (new_mean - old_mean)
            let correction_scale = ((self.n_samples_seen - n_samples) as f64 * n_samples as f64
                / self.n_samples_seen as f64)
                .sqrt();
            let correction: Vec<f64> = (0..n_features)
                .map(|f| correction_scale * (new_mean[f] - old_mean[f]))
                .collect();

            // Existing components scaled by singular values
            for i in 0..k {
                let sv = self.singular_values[i];
                let row: Vec<f64> = self.components[i].iter().map(|&v| v * sv).collect();
                augmented.push(row);
            }
            augmented.push(correction);
        }

        // Append centred batch rows
        for row in x_c {
            augmented.push(row);
        }

        // ── Thin SVD of augmented matrix (rows × n_features) ─────────────────
        let k = self.n_components.min(augmented.len()).min(n_features);
        let (u, s, vt) = thin_svd(augmented, k)?;

        self.components = vt; // k × n_features
        self.singular_values = s;

        // U is not stored (we don't need the sample projections for the model)
        let _ = u;

        Ok(())
    }

    /// Fit on an entire dataset by splitting into mini-batches.
    pub fn fit(&mut self, x: &[Vec<f64>], batch_size: usize) -> WasmResult<()> {
        if x.is_empty() {
            return Ok(());
        }
        let bs = batch_size.max(1);
        let mut start = 0;
        while start < x.len() {
            let end = (start + bs).min(x.len());
            self.partial_fit(&x[start..end])?;
            start = end;
        }
        Ok(())
    }

    /// Project `x` onto the principal components.
    ///
    /// Returns an n_samples × n_components matrix.
    pub fn transform(&self, x: &[Vec<f64>]) -> WasmResult<Vec<Vec<f64>>> {
        if self.components.is_empty() {
            return Err(WasmError::InvalidParameter(
                "Model has not been fitted yet".to_string(),
            ));
        }
        let n_features = self.n_features;
        let k = self.components.len();

        x.iter()
            .map(|row| {
                if row.len() != n_features {
                    return Err(WasmError::InvalidDimensions(format!(
                        "Expected {n_features} features, got {}",
                        row.len()
                    )));
                }
                // Centre
                let centred: Vec<f64> = row
                    .iter()
                    .enumerate()
                    .map(|(f, &v)| v - self.mean[f])
                    .collect();
                // Project: dot(centred, components^T)  → length k
                let projected: Vec<f64> = (0..k)
                    .map(|i| {
                        self.components[i]
                            .iter()
                            .zip(centred.iter())
                            .map(|(&c, &x)| c * x)
                            .sum()
                    })
                    .collect();

                if self.whiten {
                    Ok(projected
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| {
                            let sv = self.singular_values[i];
                            if sv.abs() < 1e-15 {
                                0.0
                            } else {
                                v / sv
                            }
                        })
                        .collect())
                } else {
                    Ok(projected)
                }
            })
            .collect()
    }

    /// Reconstruct samples from their projections.
    pub fn inverse_transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if self.components.is_empty() || x.is_empty() {
            return Vec::new();
        }
        let k = self.components.len();

        x.iter()
            .map(|proj| {
                let k_actual = proj.len().min(k);
                let mut rec = self.mean.clone();
                for (i, proj_val) in proj.iter().enumerate().take(k_actual) {
                    let coef = if self.whiten {
                        proj_val * self.singular_values[i]
                    } else {
                        *proj_val
                    };
                    for (rec_f, &comp_f) in rec.iter_mut().zip(self.components[i].iter()) {
                        *rec_f += coef * comp_f;
                    }
                }
                rec
            })
            .collect()
    }

    /// Fraction of variance explained by each component:
    /// σᵢ² / Σ σⱼ²
    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        if self.singular_values.is_empty() {
            return Vec::new();
        }
        let total: f64 = self.singular_values.iter().map(|&s| s * s).sum();
        if total < 1e-30 {
            return vec![0.0_f64; self.singular_values.len()];
        }
        self.singular_values
            .iter()
            .map(|&s| s * s / total)
            .collect()
    }

    /// Number of retained components
    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

// ─── Internal thin SVD ────────────────────────────────────────────────────────

/// SVD triplet: (U: m×k, S: k, Vt: k×n).
type SvdTriplet = (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>);

/// Thin SVD via QR + Gram-Schmidt orthonormalisation + power iteration.
///
/// `a` is an m × n matrix (rows are samples); returns the top-`k` singular
/// triplet (U: m×k, S: k, Vt: k×n).
pub(crate) fn thin_svd(a: Vec<Vec<f64>>, k: usize) -> WasmResult<SvdTriplet> {
    let m = a.len();
    if m == 0 {
        return Err(WasmError::InvalidParameter("Empty matrix".to_string()));
    }
    let n = a[0].len();
    if n == 0 {
        return Err(WasmError::InvalidParameter(
            "Zero-column matrix".to_string(),
        ));
    }
    let k = k.min(m).min(n);
    if k == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    // --- Compute A^T A  (n × n) ----------------------------------------------
    // For the incremental PCA use-case, m (= n_components + batch_size) is
    // typically much smaller than n_features, so we work in the m-dimensional
    // space: compute A A^T (m × m), get its eigen-decomposition, then recover V.

    if m <= n {
        // Use m×m Gram matrix  G = A A^T
        let g = mat_mul_aat(&a); // m × m
        let (eigvecs, eigvals) = power_iteration_eig(&g, k)?;
        // eigvecs: m × k, eigvals: k  (descending)

        // Singular values
        let s: Vec<f64> = eigvals
            .iter()
            .map(|&lam| if lam > 0.0 { lam.sqrt() } else { 0.0 })
            .collect();

        // U columns = eigenvectors of G
        // V columns = A^T U / σ
        let u: Vec<Vec<f64>> = (0..m)
            .map(|i| (0..k).map(|j| eigvecs[i][j]).collect())
            .collect();

        let mut vt: Vec<Vec<f64>> = Vec::with_capacity(k);
        for j in 0..k {
            // v_j = A^T u_j / s_j
            let u_col: Vec<f64> = (0..m).map(|i| eigvecs[i][j]).collect();
            let mut v_row = vec![0.0_f64; n];
            if s[j] > 1e-15 {
                // A^T u
                for (row_idx, a_row) in a.iter().enumerate() {
                    let ui = u_col[row_idx];
                    for (f, &av) in a_row.iter().enumerate() {
                        v_row[f] += ui * av;
                    }
                }
                for v in v_row.iter_mut() {
                    *v /= s[j];
                }
            }
            vt.push(v_row);
        }

        Ok((u, s, vt))
    } else {
        // n < m: work with n×n Gram matrix B = A^T A
        let b = mat_mul_ata(&a); // n × n
        let (eigvecs, eigvals) = power_iteration_eig(&b, k)?;

        let s: Vec<f64> = eigvals
            .iter()
            .map(|&lam| if lam > 0.0 { lam.sqrt() } else { 0.0 })
            .collect();

        // Vt: eigvecs of B
        let vt: Vec<Vec<f64>> = (0..k)
            .map(|j| (0..n).map(|f| eigvecs[f][j]).collect())
            .collect();

        // U: A V / S
        let mut u: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; m];
        for j in 0..k {
            if s[j] > 1e-15 {
                // A v_j
                for (i, a_row) in a.iter().enumerate() {
                    let mut val = 0.0_f64;
                    for (f, &av) in a_row.iter().enumerate() {
                        val += av * eigvecs[f][j];
                    }
                    u[i][j] = val / s[j];
                }
            }
        }

        Ok((u, s, vt))
    }
}

// ─── Matrix helpers ───────────────────────────────────────────────────────────

/// Compute A A^T (m × m) where A is m × n
fn mat_mul_aat(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let mut g = vec![vec![0.0_f64; m]; m];
    for i in 0..m {
        for j in i..m {
            let dot: f64 = a[i].iter().zip(a[j].iter()).map(|(&x, &y)| x * y).sum();
            g[i][j] = dot;
            g[j][i] = dot;
        }
    }
    g
}

/// Compute A^T A (n × n) where A is m × n
fn mat_mul_ata(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a[0].len();
    let mut b = vec![vec![0.0_f64; n]; n];
    for row in a {
        for i in 0..n {
            for j in i..n {
                let v = row[i] * row[j];
                b[i][j] += v;
                if i != j {
                    b[j][i] += v;
                }
            }
        }
    }
    b
}

/// Power iteration with deflation to find the top-k eigenvalues and eigenvectors
/// of a symmetric positive semi-definite matrix.
///
/// Returns (eigvecs: n × k column matrix, eigvals: k) in descending order.
fn power_iteration_eig(mat: &[Vec<f64>], k: usize) -> WasmResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let n = mat.len();
    if n == 0 {
        return Err(WasmError::InvalidParameter(
            "Empty matrix for eigen decomposition".to_string(),
        ));
    }
    let k = k.min(n);
    let n_iters = 200;

    let mut eigvecs: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut eigvals: Vec<f64> = Vec::with_capacity(k);
    // Residual matrix starts as a copy of mat
    let mut residual = mat.to_vec();

    for _component in 0..k {
        // Initialize random-ish vector using deterministic values
        let mut v: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
        normalize(&mut v);

        let mut lambda = 0.0_f64;
        for _ in 0..n_iters {
            // w = residual * v
            let w: Vec<f64> = (0..n)
                .map(|i| {
                    residual[i]
                        .iter()
                        .zip(v.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
                })
                .collect();
            // Rayleigh quotient
            lambda = dot(&v, &w);
            v = w;
            normalize(&mut v);
        }

        eigvecs.push(v.clone());
        eigvals.push(lambda.max(0.0));

        // Deflate: residual -= lambda * v v^T
        for i in 0..n {
            for j in 0..n {
                residual[i][j] -= lambda * v[i] * v[j];
            }
        }
    }

    Ok((
        // Convert from Vec of column vectors to column matrix n × k
        {
            let mut cols = vec![vec![0.0_f64; k]; n];
            for (j, ev) in eigvecs.iter().enumerate() {
                for (i, &val) in ev.iter().enumerate() {
                    cols[i][j] = val;
                }
            }
            cols
        },
        eigvals,
    ))
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for val in v.iter_mut() {
            *val /= norm;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate n samples with 2 dominant principal components.
    /// x = [a*cos(θ), a*sin(θ), noise] with a >> noise.
    fn make_test_data(n: usize, n_features: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let theta = 2.0 * PI * i as f64 / n as f64;
                let mut row = vec![0.0_f64; n_features];
                row[0] = 3.0 * theta.cos();
                row[1] = 2.0 * theta.sin();
                for (f, cell) in row.iter_mut().enumerate().skip(2) {
                    // Small noise-like variation
                    *cell = 0.05 * (i as f64 * 0.1 + f as f64).sin();
                }
                row
            })
            .collect()
    }

    #[test]
    fn test_partial_fit_dimensions() {
        let n_features = 8;
        let n_components = 3;
        let mut ipca = IncrementalPca::new(n_components, n_features);
        let data = make_test_data(50, n_features);
        ipca.partial_fit(&data).expect("partial_fit failed");

        assert_eq!(ipca.components.len(), n_components);
        assert_eq!(ipca.singular_values.len(), n_components);
        for row in &ipca.components {
            assert_eq!(row.len(), n_features);
        }
    }

    #[test]
    fn test_fit_in_batches() {
        let n_features = 6;
        let n_components = 3;
        let mut ipca = IncrementalPca::new(n_components, n_features);
        let data = make_test_data(200, n_features);
        ipca.fit(&data, 50).expect("fit failed");

        assert_eq!(ipca.components.len(), n_components);
        assert_eq!(ipca.n_samples_seen, 200);
    }

    #[test]
    fn test_transform_shape() {
        let n_features = 8;
        let n_components = 3;
        let mut ipca = IncrementalPca::new(n_components, n_features);
        let train = make_test_data(100, n_features);
        ipca.fit(&train, 50).expect("fit");

        let test = make_test_data(20, n_features);
        let proj = ipca.transform(&test).expect("transform");
        assert_eq!(proj.len(), 20);
        for row in &proj {
            assert_eq!(row.len(), n_components);
        }
    }

    #[test]
    fn test_explained_variance_ratio_sums_near_1() {
        let n_features = 6;
        let n_components = 6; // All components
        let mut ipca = IncrementalPca::new(n_components, n_features);
        let data = make_test_data(150, n_features);
        ipca.fit(&data, 75).expect("fit");

        let evr = ipca.explained_variance_ratio();
        assert!(!evr.is_empty());
        let total: f64 = evr.iter().sum();
        assert!((total - 1.0).abs() < 0.01, "EVR sum = {total}");
    }

    #[test]
    fn test_inverse_transform_reconstruction() {
        let n_features = 6;
        let n_components = 4;
        let mut ipca = IncrementalPca::new(n_components, n_features);
        let data = make_test_data(200, n_features);
        ipca.fit(&data, 100).expect("fit");

        let proj = ipca.transform(&data[..10]).expect("transform");
        let rec = ipca.inverse_transform(&proj);

        assert_eq!(rec.len(), 10);
        // Reconstruction won't be perfect (only 4 components), but should be close
        // for the first two features which have most variance.
        for (orig, recon) in data[..10].iter().zip(rec.iter()) {
            let err: f64 = orig
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(err < 3.0, "Reconstruction error too high: {err}");
        }
    }

    #[test]
    fn test_incremental_vs_single_batch() {
        // Fitting in one batch vs two batches of 100 should give similar EVR
        let n_features = 6;
        let n_components = 3;
        let data = make_test_data(200, n_features);

        let mut ipca1 = IncrementalPca::new(n_components, n_features);
        ipca1.partial_fit(&data).expect("single batch");

        let mut ipca2 = IncrementalPca::new(n_components, n_features);
        ipca2.partial_fit(&data[..100]).expect("batch 1");
        ipca2.partial_fit(&data[100..]).expect("batch 2");

        let evr1 = ipca1.explained_variance_ratio();
        let evr2 = ipca2.explained_variance_ratio();

        // First component should dominate in both
        assert!(
            evr1[0] > 0.3 && evr2[0] > 0.3,
            "EVR1[0]={}, EVR2[0]={}",
            evr1[0],
            evr2[0]
        );
    }

    #[test]
    fn test_transform_requires_fit() {
        let ipca = IncrementalPca::new(3, 5);
        let data = make_test_data(5, 5);
        let result = ipca.transform(&data);
        assert!(result.is_err(), "transform should fail before fit");
    }

    #[test]
    fn test_n_components_accessor() {
        let ipca = IncrementalPca::new(7, 10);
        assert_eq!(ipca.n_components(), 7);
    }
}
