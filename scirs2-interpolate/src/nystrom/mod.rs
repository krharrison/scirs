//! Nyström approximation for large-scale Kriging.
//!
//! Large-scale kernel methods are made tractable by approximating the full
//! n×n kernel matrix K with a low-rank approximation built from m « n
//! landmark points:
//!
//! ```text
//! K ≈ K_nm · K_mm^{-1} · K_nm^T
//! ```
//!
//! where `K_nm[i,j] = k(x_i, x_j^*)` with x^* being the m landmark points.
//!
//! # References
//! - Williams, C. K. I. & Seeger, M. (2001). Using the Nyström method to speed up
//!   kernel machines. NIPS.

use crate::error::InterpolateError;
use crate::random_features::KernelType;

// ─── NystromConfig ───────────────────────────────────────────────────────────

/// Configuration for Nyström Kriging.
#[derive(Debug, Clone)]
pub struct NystromConfig {
    /// Number of landmark (inducing) points.
    pub n_landmarks: usize,
    /// Kernel type.
    pub kernel: KernelType,
    /// Kernel bandwidth / length-scale.
    pub bandwidth: f64,
    /// Nugget / regularization added to the diagonal.
    pub regularization: f64,
    /// Seed for landmark selection.
    pub seed: u64,
}

impl Default for NystromConfig {
    fn default() -> Self {
        Self {
            n_landmarks: 100,
            kernel: KernelType::Gaussian,
            bandwidth: 1.0,
            regularization: 1e-6,
            seed: 0,
        }
    }
}

// ─── Kernel evaluation ───────────────────────────────────────────────────────

/// Evaluate a single kernel value k(x1, x2).
pub fn apply_kernel(kernel: &KernelType, bandwidth: f64, x1: &[f64], x2: &[f64]) -> f64 {
    match kernel {
        KernelType::Gaussian => {
            let sq: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            (-sq / (2.0 * bandwidth * bandwidth)).exp()
        }
        KernelType::Laplacian => {
            let l1: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).abs()).sum();
            (-l1 / bandwidth).exp()
        }
        KernelType::Cauchy => {
            let sq: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            1.0 / (1.0 + sq / (bandwidth * bandwidth))
        }
        KernelType::Matern32 => {
            let r: f64 = x1
                .iter()
                .zip(x2.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let v = 3.0_f64.sqrt() * r / bandwidth;
            (1.0 + v) * (-v).exp()
        }
        KernelType::Matern52 => {
            let r: f64 = x1
                .iter()
                .zip(x2.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let v = 5.0_f64.sqrt() * r / bandwidth;
            (1.0 + v + v * v / 3.0) * (-v).exp()
        }
    }
}

// ─── NystromKriging ──────────────────────────────────────────────────────────

/// Nyström-approximated Gaussian process / Kriging model.
#[derive(Debug, Clone)]
pub struct NystromKriging {
    /// Configuration.
    pub config: NystromConfig,
    /// Selected landmark points, shape `[m][d]`.
    pub landmarks: Vec<Vec<f64>>,
    /// Kernel matrix between training points and landmarks K_nm `[n][m]`.
    pub k_nm: Vec<Vec<f64>>,
    /// Inverse of K_mm, shape `[m][m]`.
    pub k_mm_inv: Vec<Vec<f64>>,
    /// Dual coefficients α (solution to the system), shape `[n]`.
    pub alpha: Vec<f64>,
    /// Whether the model has been fitted.
    fitted: bool,
}

impl NystromKriging {
    /// Create a new (unfitted) Nyström Kriging model.
    pub fn new(config: NystromConfig) -> Self {
        Self {
            config,
            landmarks: Vec::new(),
            k_nm: Vec::new(),
            k_mm_inv: Vec::new(),
            alpha: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the model to training data.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), InterpolateError> {
        let n = x.len();
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "Training data is empty".to_string(),
            ));
        }
        if n != y.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "x has {} rows but y has {} elements",
                n,
                y.len()
            )));
        }
        let m = self.config.n_landmarks.min(n);

        // ── Select landmarks via k-means++ seeding ──────────────────────────
        self.landmarks = select_landmarks_kmeanspp(x, m, self.config.seed)?;

        let d = self.landmarks.len(); // actual m (may be < n_landmarks if n is small)

        // ── Build K_mm (m × m) ───────────────────────────────────────────────
        let k_mm = build_kernel_matrix(
            &self.landmarks,
            &self.landmarks,
            &self.config.kernel,
            self.config.bandwidth,
        );

        // Regularise K_mm for inversion stability
        let mut k_mm_reg = k_mm.clone();
        for i in 0..d {
            k_mm_reg[i][i] += self.config.regularization;
        }
        self.k_mm_inv = mat_inv(&k_mm_reg)?;

        // ── Build K_nm (n × m) ───────────────────────────────────────────────
        self.k_nm = build_kernel_matrix(
            x,
            &self.landmarks,
            &self.config.kernel,
            self.config.bandwidth,
        );

        // ── Nyström approximation K ≈ K_nm K_mm^{-1} K_nm^T ─────────────────
        // Build (K_nm K_mm^{-1} K_nm^T + λI) and solve for α using CG.
        // We avoid forming n×n explicitly — use matrix-vector products.

        let lambda = self.config.regularization;
        let rhs = y.to_vec();

        // mv(v): (K_nm K_mm^{-1} K_nm^T + λI) v
        let mv = |v: &[f64]| -> Vec<f64> {
            // Step 1: t1 = K_nm^T v  (m-vector)
            let t1 = mat_vec_mul_t(&self.k_nm, v); // m
                                                   // Step 2: t2 = K_mm^{-1} t1  (m-vector)
            let t2 = mat_vec_mul(&self.k_mm_inv, &t1); // m
                                                       // Step 3: result = K_nm t2 + λ v  (n-vector)
            let mut res = mat_vec_mul(&self.k_nm, &t2);
            for (r, vi) in res.iter_mut().zip(v.iter()) {
                *r += lambda * vi;
            }
            res
        };

        self.alpha = conjugate_gradient(mv, &rhs, 200)?;
        self.fitted = true;
        Ok(())
    }

    /// Predict at test points.
    pub fn predict(&self, x_test: &[Vec<f64>]) -> Result<Vec<f64>, InterpolateError> {
        if !self.fitted {
            return Err(InterpolateError::InvalidState(
                "Model not fitted yet".to_string(),
            ));
        }
        let k_test_m = build_kernel_matrix(
            x_test,
            &self.landmarks,
            &self.config.kernel,
            self.config.bandwidth,
        );

        // y_pred = K_test_m K_mm^{-1} K_nm^T α
        // = K_test_m K_mm^{-1} (K_nm^T α)
        let knt_alpha = mat_vec_mul_t(&self.k_nm, &self.alpha);
        let km_inv_knt_alpha = mat_vec_mul(&self.k_mm_inv, &knt_alpha);
        let y_pred = mat_vec_mul(&k_test_m, &km_inv_knt_alpha);
        Ok(y_pred)
    }

    /// Predict with posterior variance estimate.
    ///
    /// Returns `(mean, variance)` where variance uses the Nyström approximation:
    /// ```text
    /// σ²(x*) = k(x*,x*) - k_m(x*)^T K_mm^{-1} k_m(x*)
    /// ```
    pub fn predict_with_variance(
        &self,
        x_test: &[Vec<f64>],
    ) -> Result<(Vec<f64>, Vec<f64>), InterpolateError> {
        let means = self.predict(x_test)?;

        let n_test = x_test.len();
        let mut variances = Vec::with_capacity(n_test);

        for xi in x_test.iter() {
            // k(x*,x*) = 1 for standard kernels at self
            let k_self = apply_kernel(&self.config.kernel, self.config.bandwidth, xi, xi);

            // k_m(x*): kernel between x* and each landmark
            let k_m: Vec<f64> = self
                .landmarks
                .iter()
                .map(|lm| apply_kernel(&self.config.kernel, self.config.bandwidth, xi, lm))
                .collect();

            // K_mm^{-1} k_m(x*)
            let km_inv_km = mat_vec_mul(&self.k_mm_inv, &k_m);

            // k_m^T K_mm^{-1} k_m
            let quad: f64 = k_m.iter().zip(km_inv_km.iter()).map(|(a, b)| a * b).sum();

            let var = (k_self - quad).max(0.0);
            variances.push(var);
        }

        Ok((means, variances))
    }
}

// ─── Helper: landmark selection via k-means++ seeding ───────────────────────

fn select_landmarks_kmeanspp(
    x: &[Vec<f64>],
    m: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, InterpolateError> {
    let n = x.len();
    if m >= n {
        return Ok(x.to_vec());
    }

    let mut rng = SimpleLcg::new(seed.wrapping_add(17));
    let mut chosen: Vec<usize> = Vec::with_capacity(m);

    // Random first center
    let first = (rng.next_f64() * n as f64) as usize % n;
    chosen.push(first);

    // k-means++ seeding: choose next center proportional to distance²
    for _ in 1..m {
        let dists: Vec<f64> = (0..n)
            .map(|i| {
                if chosen.contains(&i) {
                    return 0.0;
                }
                chosen
                    .iter()
                    .map(|&c| sq_dist(&x[i], &x[c]))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        let total: f64 = dists.iter().sum();
        if total < 1e-300 {
            break;
        }
        let mut thresh = rng.next_f64() * total;
        let mut next_idx = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            thresh -= d;
            if thresh <= 0.0 {
                next_idx = i;
                break;
            }
        }
        if !chosen.contains(&next_idx) {
            chosen.push(next_idx);
        }
    }

    Ok(chosen.into_iter().map(|i| x[i].clone()).collect())
}

// ─── Helper: matrix operations ───────────────────────────────────────────────

/// Build kernel matrix `K[i][j] = k(a[i], b[j])`.
fn build_kernel_matrix(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    kernel: &KernelType,
    bandwidth: f64,
) -> Vec<Vec<f64>> {
    a.iter()
        .map(|ai| {
            b.iter()
                .map(|bj| apply_kernel(kernel, bandwidth, ai, bj))
                .collect()
        })
        .collect()
}

/// Matrix-vector product y = A * x.
fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Matrix-vector product with transpose y = A^T * x.
fn mat_vec_mul_t(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a[0].len();
    let mut result = vec![0.0f64; m];
    for (i, row) in a.iter().enumerate() {
        if i < x.len() {
            for (j, &aij) in row.iter().enumerate() {
                result[j] += aij * x[i];
            }
        }
    }
    result
}

/// Invert a small symmetric positive definite matrix via Cholesky.
fn mat_inv(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, InterpolateError> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Augmented matrix [A | I]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            let mut e = vec![0.0f64; n];
            e[i] = 1.0;
            r.extend(e);
            r
        })
        .collect();

    // Gauss-Jordan elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-300 {
            return Err(InterpolateError::LinalgError(
                "Singular matrix in Nyström inversion".to_string(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    let val = factor * aug[col][j];
                    aug[row][j] -= val;
                }
            }
        }
    }

    Ok(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

/// Conjugate gradient solver for A*x = b where A is SPD.
fn conjugate_gradient(
    a_mv: impl Fn(&[f64]) -> Vec<f64>,
    b: &[f64],
    max_iter: usize,
) -> Result<Vec<f64>, InterpolateError> {
    let n = b.len();
    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old: f64 = r.iter().map(|v| v * v).sum();

    if rs_old.sqrt() < 1e-14 {
        return Ok(x);
    }

    for _ in 0..max_iter {
        let ap = a_mv(&p);
        let pap: f64 = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-300 {
            break;
        }
        let alpha = rs_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new: f64 = r.iter().map(|v| v * v).sum();
        if rs_new.sqrt() < 1e-10 {
            break;
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Ok(x)
}

/// Squared Euclidean distance.
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum()
}

// ─── Simple LCG (local copy to avoid import cycle) ───────────────────────────
struct SimpleLcg {
    state: u64,
}

impl SimpleLcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gaussian_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut lcg = SimpleLcg::new(seed);
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| vec![lcg.next_f64() * 4.0 - 2.0, lcg.next_f64() * 4.0 - 2.0])
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|xi| (-xi[0] * xi[0] - xi[1] * xi[1]).exp())
            .collect();
        (x, y)
    }

    #[test]
    fn test_nystrom_fit_and_predict() {
        let (x, y) = make_gaussian_data(50, 42);
        let config = NystromConfig {
            n_landmarks: 15,
            kernel: KernelType::Gaussian,
            bandwidth: 1.0,
            regularization: 1e-4,
            seed: 0,
        };
        let mut model = NystromKriging::new(config);
        model.fit(&x, &y).expect("fit");

        let preds = model.predict(&x).expect("predict");
        assert_eq!(preds.len(), x.len());

        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, yi)| (p - yi).powi(2))
            .sum::<f64>()
            / preds.len() as f64;
        assert!(mse < 0.5, "MSE too large: {mse}");
    }

    #[test]
    fn test_predict_variance_non_negative() {
        let (x, y) = make_gaussian_data(30, 77);
        let config = NystromConfig {
            n_landmarks: 10,
            kernel: KernelType::Gaussian,
            bandwidth: 1.0,
            regularization: 1e-4,
            seed: 1,
        };
        let mut model = NystromKriging::new(config);
        model.fit(&x, &y).expect("fit");

        let x_test: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![i as f64 * 0.3, i as f64 * 0.3])
            .collect();
        let (means, vars) = model
            .predict_with_variance(&x_test)
            .expect("predict_with_variance");
        assert_eq!(means.len(), 5);
        assert_eq!(vars.len(), 5);
        for v in &vars {
            assert!(*v >= 0.0, "Variance must be non-negative, got {v}");
        }
    }

    #[test]
    fn test_apply_kernel_values() {
        let x1 = vec![0.0f64, 0.0];
        let x2 = vec![1.0f64, 0.0];
        let bw = 1.0;

        // Gaussian: exp(-0.5)
        let g = apply_kernel(&KernelType::Gaussian, bw, &x1, &x2);
        let expected = (-0.5f64).exp();
        assert!((g - expected).abs() < 1e-10, "Gaussian kernel mismatch");

        // Laplacian: exp(-1)
        let l = apply_kernel(&KernelType::Laplacian, bw, &x1, &x2);
        let exp_l = (-1.0f64).exp();
        assert!((l - exp_l).abs() < 1e-10, "Laplacian kernel mismatch");

        // Matern32: (1 + sqrt(3)) * exp(-sqrt(3))
        let m32 = apply_kernel(&KernelType::Matern32, bw, &x1, &x2);
        let v = 3.0_f64.sqrt();
        let expected_m32 = (1.0 + v) * (-v).exp();
        assert!(
            (m32 - expected_m32).abs() < 1e-10,
            "Matern32 kernel mismatch"
        );
    }

    #[test]
    fn test_nystrom_matern_kernel() {
        let (x, y) = make_gaussian_data(20, 100);
        let config = NystromConfig {
            n_landmarks: 8,
            kernel: KernelType::Matern52,
            bandwidth: 1.5,
            regularization: 1e-4,
            seed: 5,
        };
        let mut model = NystromKriging::new(config);
        model.fit(&x, &y).expect("fit matern52");
        let preds = model.predict(&x[..5]).expect("predict");
        assert_eq!(preds.len(), 5);
        for p in &preds {
            assert!(p.is_finite(), "Prediction should be finite");
        }
    }
}
