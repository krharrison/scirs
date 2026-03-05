//! Multi-View Dimensionality Reduction
//!
//! This module implements methods that jointly embed or correlate multiple
//! views (feature sets) of the same data.
//!
//! ## Algorithms
//!
//! - [`CCA`]: Canonical Correlation Analysis — find maximally correlated linear
//!   projections of two views via symmetric eigenvalue decomposition.
//! - [`KCCA`]: Kernel CCA — extend CCA to nonlinear mappings via kernel trick.
//! - [`GCCA`]: Generalized CCA (SUM-SQ objective) — handle more than two views.
//!
//! ## References
//!
//! - Hardoon, D.R., Szedmak, S., & Shawe-Taylor, J. (2004). Canonical
//!   Correlation Analysis: An Overview with Application to Learning Methods.
//!   Neural Computation.
//! - Kettenring, J.R. (1971). Canonical Analysis of Several Sets of Variables.
//!   Biometrika.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::{eigh, inv, svd};
// UPLO removed: eigh now takes Option<usize> workers parameter

use crate::error::{Result, TransformError};

// ============================================================================
// Helper utilities
// ============================================================================

/// Compute the sample covariance matrix of `x` (n × p).
/// Returns a p × p matrix.
fn sample_cov(x: &[Vec<f64>]) -> Result<Array2<f64>> {
    let n = x.len();
    if n < 2 {
        return Err(TransformError::InvalidInput(
            "Need at least 2 samples for covariance".to_string(),
        ));
    }
    let p = x[0].len();
    if p == 0 {
        return Err(TransformError::InvalidInput(
            "Feature dimension must be > 0".to_string(),
        ));
    }

    // Column means
    let mut means = vec![0.0f64; p];
    for row in x.iter() {
        if row.len() != p {
            return Err(TransformError::InvalidInput(
                "All rows must have the same length".to_string(),
            ));
        }
        for (j, &v) in row.iter().enumerate() {
            means[j] += v;
        }
    }
    for m in means.iter_mut() {
        *m /= n as f64;
    }

    // Covariance
    let mut cov = Array2::<f64>::zeros((p, p));
    for row in x.iter() {
        for i in 0..p {
            let ci = row[i] - means[i];
            for j in 0..p {
                let cj = row[j] - means[j];
                cov[[i, j]] += ci * cj;
            }
        }
    }
    let scale = 1.0 / (n as f64 - 1.0);
    cov.mapv_inplace(|v| v * scale);
    Ok(cov)
}

/// Compute the cross-covariance matrix of x (n×p) and y (n×q).
/// Returns a p × q matrix.
fn cross_cov(x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<Array2<f64>> {
    let n = x.len();
    if n != y.len() {
        return Err(TransformError::InvalidInput(
            "x and y must have the same number of rows".to_string(),
        ));
    }
    if n < 2 {
        return Err(TransformError::InvalidInput(
            "Need at least 2 samples".to_string(),
        ));
    }
    let p = x[0].len();
    let q = y[0].len();

    let mut xmeans = vec![0.0f64; p];
    let mut ymeans = vec![0.0f64; q];
    for (xrow, yrow) in x.iter().zip(y.iter()) {
        if xrow.len() != p {
            return Err(TransformError::InvalidInput(
                "All x-rows must have the same length".to_string(),
            ));
        }
        if yrow.len() != q {
            return Err(TransformError::InvalidInput(
                "All y-rows must have the same length".to_string(),
            ));
        }
        for (j, &v) in xrow.iter().enumerate() {
            xmeans[j] += v;
        }
        for (j, &v) in yrow.iter().enumerate() {
            ymeans[j] += v;
        }
    }
    for m in xmeans.iter_mut() {
        *m /= n as f64;
    }
    for m in ymeans.iter_mut() {
        *m /= n as f64;
    }

    let mut cc = Array2::<f64>::zeros((p, q));
    for (xrow, yrow) in x.iter().zip(y.iter()) {
        for i in 0..p {
            let ci = xrow[i] - xmeans[i];
            for j in 0..q {
                let cj = yrow[j] - ymeans[j];
                cc[[i, j]] += ci * cj;
            }
        }
    }
    let scale = 1.0 / (n as f64 - 1.0);
    cc.mapv_inplace(|v| v * scale);
    Ok(cc)
}

/// Compute column means of a dataset.
fn col_means(x: &[Vec<f64>]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    let p = x[0].len();
    let mut means = vec![0.0f64; p];
    for row in x.iter() {
        for (j, &v) in row.iter().enumerate() {
            means[j] += v;
        }
    }
    let n = x.len() as f64;
    for m in means.iter_mut() {
        *m /= n;
    }
    means
}

/// Add regularization to the diagonal of a matrix (in-place).
fn add_ridge(a: &mut Array2<f64>, reg: f64) {
    let n = a.nrows().min(a.ncols());
    for i in 0..n {
        a[[i, i]] += reg;
    }
}

/// Compute A^{-1/2} via eigendecomposition: A = V D V^T => A^{-1/2} = V D^{-1/2} V^T.
fn matrix_inv_sqrt(a: &Array2<f64>, reg: f64) -> Result<Array2<f64>> {
    let mut a_reg = a.clone();
    add_ridge(&mut a_reg, reg);
    let (eigenvalues, eigenvectors) =
        eigh(&a_reg.view(), None).map_err(|e| {
            TransformError::ComputationError(format!("eigh failed in inv_sqrt: {e}"))
        })?;

    let p = a.nrows();
    let mut d_inv_sqrt = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        let ev = eigenvalues[i].max(reg);
        d_inv_sqrt[[i, i]] = 1.0 / ev.sqrt();
    }

    // A^{-1/2} = V * D^{-1/2} * V^T
    let vd = eigenvectors.dot(&d_inv_sqrt);
    Ok(vd.dot(&eigenvectors.t()))
}

// ============================================================================
// CCA — Canonical Correlation Analysis
// ============================================================================

/// Configuration for Canonical Correlation Analysis.
#[derive(Debug, Clone)]
pub struct CCA {
    /// Number of canonical components to extract.
    pub n_components: usize,
    /// Regularization added to covariance matrices for numerical stability.
    pub reg: f64,
}

/// Fitted CCA model storing projection matrices and canonical correlations.
#[derive(Debug, Clone)]
pub struct CCAModel {
    /// n_components
    pub n_components: usize,
    /// Projection for X: shape (p, n_components)
    pub wx: Array2<f64>,
    /// Projection for Y: shape (q, n_components)
    pub wy: Array2<f64>,
    /// Canonical correlations: length n_components
    pub correlations: Vec<f64>,
    /// Column means of X used during fit
    pub x_mean: Vec<f64>,
    /// Column means of Y used during fit
    pub y_mean: Vec<f64>,
}

impl Default for CCA {
    fn default() -> Self {
        CCA {
            n_components: 2,
            reg: 1e-6,
        }
    }
}

impl CCA {
    /// Create a new CCA with the given number of components and regularization.
    pub fn new(n_components: usize, reg: f64) -> Self {
        CCA { n_components, reg }
    }

    /// Fit CCA on views X (n×p) and Y (n×q).
    ///
    /// Uses the SVD approach on the whitened cross-covariance matrix:
    /// T = C_XX^{-1/2} · C_XY · C_YY^{-1/2}
    /// SVD(T) = U Σ V^T  →  wx = C_XX^{-1/2} U,  wy = C_YY^{-1/2} V
    pub fn fit(&self, x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<CCAModel> {
        let n = x.len();
        if n == 0 {
            return Err(TransformError::InvalidInput("Empty dataset".to_string()));
        }
        if n != y.len() {
            return Err(TransformError::InvalidInput(
                "X and Y must have equal number of samples".to_string(),
            ));
        }

        let p = x[0].len();
        let q = y[0].len();
        let k = self.n_components.min(p).min(q);
        if k == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }

        let x_mean = col_means(x);
        let y_mean = col_means(y);

        let cxx = sample_cov(x)?;
        let cyy = sample_cov(y)?;
        let cxy = cross_cov(x, y)?;

        // Whitening matrices
        let cxx_inv_sqrt = matrix_inv_sqrt(&cxx, self.reg)?;
        let cyy_inv_sqrt = matrix_inv_sqrt(&cyy, self.reg)?;

        // T = C_XX^{-1/2} C_XY C_YY^{-1/2}
        let t = cxx_inv_sqrt.dot(&cxy).dot(&cyy_inv_sqrt);

        // SVD of T
        let (u_mat, sigma, vt_mat) = svd(&t.view(), true, None)
            .map_err(|e| TransformError::ComputationError(format!("SVD failed in CCA: {e}")))?;

        // Take first k components
        let u_k = u_mat.slice(scirs2_core::ndarray::s![.., ..k]).to_owned();
        let vt_k = vt_mat.slice(scirs2_core::ndarray::s![..k, ..]).to_owned();

        let correlations: Vec<f64> = (0..k).map(|i| sigma[i].min(1.0).max(-1.0)).collect();

        // wx = C_XX^{-1/2} U_k  (p × k)
        let wx = cxx_inv_sqrt.dot(&u_k);
        // wy = C_YY^{-1/2} V_k  (q × k) — note V = Vt^T
        let wy = cyy_inv_sqrt.dot(&vt_k.t().to_owned());

        Ok(CCAModel {
            n_components: k,
            wx,
            wy,
            correlations,
            x_mean,
            y_mean,
        })
    }
}

impl CCAModel {
    /// Project X onto the canonical directions.
    pub fn transform_x(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = x.len();
        let p = self.wx.nrows();
        let k = self.n_components;

        let mut out = vec![vec![0.0f64; k]; n];
        for (i, row) in x.iter().enumerate() {
            if row.len() != p {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i}: expected {p} features, got {}",
                    row.len()
                )));
            }
            // Center
            let centered: Vec<f64> = row.iter().zip(self.x_mean.iter()).map(|(v, m)| v - m).collect();
            for j in 0..k {
                let s: f64 = centered.iter().enumerate().map(|(fi, &cv)| cv * self.wx[[fi, j]]).sum();
                out[i][j] = s;
            }
        }
        Ok(out)
    }

    /// Project Y onto the canonical directions.
    pub fn transform_y(&self, y: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = y.len();
        let q = self.wy.nrows();
        let k = self.n_components;

        let mut out = vec![vec![0.0f64; k]; n];
        for (i, row) in y.iter().enumerate() {
            if row.len() != q {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i}: expected {q} features, got {}",
                    row.len()
                )));
            }
            let centered: Vec<f64> = row.iter().zip(self.y_mean.iter()).map(|(v, m)| v - m).collect();
            for j in 0..k {
                let s: f64 = centered.iter().enumerate().map(|(fi, &cv)| cv * self.wy[[fi, j]]).sum();
                out[i][j] = s;
            }
        }
        Ok(out)
    }

    /// Return the canonical correlations (ordered by magnitude).
    pub fn canonical_correlations(&self) -> &[f64] {
        &self.correlations
    }
}

// ============================================================================
// Kernel type for KCCA
// ============================================================================

/// Kernel function type for Kernel CCA.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    /// Linear kernel: k(x,y) = x^T y
    Linear,
    /// Polynomial kernel: k(x,y) = (gamma * x^T y + coef0)^degree
    Polynomial {
        /// Degree of the polynomial kernel
        degree: u32,
        /// Scaling factor for the dot product
        gamma: f64,
        /// Constant offset term
        coef0: f64,
    },
    /// Radial basis function: k(x,y) = exp(-gamma * ||x-y||^2)
    RBF {
        /// Width parameter controlling the Gaussian spread
        gamma: f64,
    },
    /// Sigmoid kernel: k(x,y) = tanh(gamma * x^T y + coef0)
    Sigmoid {
        /// Scaling factor for the dot product
        gamma: f64,
        /// Constant offset term
        coef0: f64,
    },
}

impl KernelType {
    fn compute(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            KernelType::Linear => a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum(),
            KernelType::Polynomial { degree, gamma, coef0 } => {
                let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
                (gamma * dot + coef0).powi(*degree as i32)
            }
            KernelType::RBF { gamma } => {
                let sq_dist: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum();
                (-gamma * sq_dist).exp()
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                let dot: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
                (gamma * dot + coef0).tanh()
            }
        }
    }
}

/// Compute the n×n kernel (Gram) matrix for a dataset.
fn kernel_matrix(x: &[Vec<f64>], kernel: &KernelType) -> Array2<f64> {
    let n = x.len();
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in i..n {
            let v = kernel.compute(&x[i], &x[j]);
            k[[i, j]] = v;
            k[[j, i]] = v;
        }
    }
    k
}

/// Center a kernel matrix: K_c = (I - 1/n 11^T) K (I - 1/n 11^T)
fn center_kernel(k: &Array2<f64>) -> Array2<f64> {
    let n = k.nrows();
    let nf = n as f64;
    let row_means = k.mean_axis(Axis(1)).unwrap_or_else(|| Array1::zeros(n));
    let grand_mean = row_means.sum() / nf;

    let mut kc = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            kc[[i, j]] = k[[i, j]] - row_means[i] - row_means[j] + grand_mean;
        }
    }
    kc
}

// ============================================================================
// KCCA — Kernel CCA
// ============================================================================

/// Kernel Canonical Correlation Analysis.
///
/// Solves the regularized dual problem in kernel space:
/// find alpha, beta maximizing the correlation between K_X alpha and K_Y beta,
/// using the regularized eigenvalue formulation.
#[derive(Debug, Clone)]
pub struct KCCA {
    /// Number of canonical components.
    pub n_components: usize,
    /// Kernel for view X.
    pub kernel_x: KernelType,
    /// Kernel for view Y.
    pub kernel_y: KernelType,
    /// Regularization parameter (added to kernel matrices).
    pub reg: f64,
}

/// Fitted KCCA model.
#[derive(Debug, Clone)]
pub struct KCCAModel {
    /// Number of components.
    pub n_components: usize,
    /// Dual coefficients for X: shape (n_train, n_components)
    pub alpha: Array2<f64>,
    /// Dual coefficients for Y: shape (n_train, n_components)
    pub beta: Array2<f64>,
    /// Canonical correlations.
    pub correlations: Vec<f64>,
    /// Training data X (needed for prediction).
    pub x_train: Vec<Vec<f64>>,
    /// Training data Y (needed for prediction).
    pub y_train: Vec<Vec<f64>>,
    /// Kernel for X.
    pub kernel_x: KernelType,
    /// Kernel for Y.
    pub kernel_y: KernelType,
}

impl Default for KCCA {
    fn default() -> Self {
        KCCA {
            n_components: 2,
            kernel_x: KernelType::RBF { gamma: 1.0 },
            kernel_y: KernelType::RBF { gamma: 1.0 },
            reg: 1e-4,
        }
    }
}

impl KCCA {
    /// Create a new KCCA instance.
    pub fn new(
        n_components: usize,
        kernel_x: KernelType,
        kernel_y: KernelType,
        reg: f64,
    ) -> Self {
        KCCA { n_components, kernel_x, kernel_y, reg }
    }

    /// Fit KCCA on two views.
    ///
    /// Uses a deflation-based approach that avoids full eigendecomposition:
    /// 1. Compute regularized inverse of kernel matrices via `inv`.
    /// 2. Form the product matrix M = (K_X + rI)^{-1} K_X K_Y (K_Y + rI)^{-1} K_Y.
    /// 3. Extract top-k directions via power iteration with deflation.
    pub fn fit(&self, x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<KCCAModel> {
        let n = x.len();
        if n == 0 || n != y.len() {
            return Err(TransformError::InvalidInput(
                "Views must have equal non-zero number of samples".to_string(),
            ));
        }
        let k = self.n_components.min(n);

        let kx = kernel_matrix(x, &self.kernel_x);
        let ky = kernel_matrix(y, &self.kernel_y);
        let kx_c = center_kernel(&kx);
        let ky_c = center_kernel(&ky);

        // Regularized kernel matrices
        let mut kx_reg = kx_c.clone();
        let mut ky_reg = ky_c.clone();
        add_ridge(&mut kx_reg, self.reg * n as f64);
        add_ridge(&mut ky_reg, self.reg * n as f64);

        // Compute inverses of regularized kernel matrices
        let kx_inv = inv(&kx_reg.view(), None)
            .map_err(|e| TransformError::ComputationError(format!("KCCA: inv(K_X+rI) failed: {e}")))?;
        let ky_inv = inv(&ky_reg.view(), None)
            .map_err(|e| TransformError::ComputationError(format!("KCCA: inv(K_Y+rI) failed: {e}")))?;

        // Form the matrix for alpha: M_x = (K_X+rI)^{-1} K_Xc K_Yc (K_Y+rI)^{-1} K_Yc
        // And for beta:              M_y = (K_Y+rI)^{-1} K_Yc K_Xc (K_X+rI)^{-1} K_Xc
        // We extract top-k directions of M_x via power iteration with deflation.
        let a_mat = kx_inv.dot(&kx_c);  // (K_X+rI)^{-1} K_Xc
        let b_mat = ky_inv.dot(&ky_c);  // (K_Y+rI)^{-1} K_Yc
        // M_x = A_mat @ K_Yc @ B_mat
        let m_x = a_mat.dot(&ky_c).dot(&b_mat);

        // Power iteration with deflation to extract top-k eigenvectors of M_x
        let max_power_iter = 200;
        let tol = 1e-8;
        let mut alphas = Array2::<f64>::zeros((n, k));
        let mut correlations = Vec::with_capacity(k);
        let mut m_deflated = m_x.clone();

        for comp in 0..k {
            // Initialize with a simple vector
            let mut v = Array1::<f64>::zeros(n);
            for i in 0..n {
                v[i] = ((i + comp + 1) as f64 * 0.618).sin();
            }
            // Normalize
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-14 {
                v.mapv_inplace(|x| x / norm);
            }

            let mut eigenvalue = 0.0_f64;
            for _iter in 0..max_power_iter {
                // v_new = M @ v
                let v_new = m_deflated.dot(&v);
                let new_norm: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
                if new_norm < 1e-14 {
                    break;
                }
                let new_eigenvalue = v.dot(&v_new);  // Rayleigh quotient
                let v_normed = &v_new / new_norm;

                // Convergence check
                let diff: f64 = v_normed.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                v = v_normed;
                eigenvalue = new_eigenvalue;
                if diff < tol {
                    break;
                }
            }

            // Store alpha
            let alpha_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if alpha_norm > 1e-14 {
                for i in 0..n {
                    alphas[[i, comp]] = v[i] / alpha_norm;
                }
            }
            correlations.push(eigenvalue.abs().sqrt().min(1.0));

            // Deflate: M_deflated -= eigenvalue * v * v^T
            for i in 0..n {
                for j in 0..n {
                    m_deflated[[i, j]] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        // Compute beta from alpha: beta_comp ~ (K_Y+rI)^{-1} K_Yc @ alpha_comp (unnormalized)
        let mut betas = Array2::<f64>::zeros((n, k));
        for comp in 0..k {
            let alpha_col = alphas.column(comp).to_owned();
            let beta_col = b_mat.dot(&kx_c.dot(&alpha_col));
            let beta_norm: f64 = beta_col.iter().map(|x| x * x).sum::<f64>().sqrt();
            if beta_norm > 1e-14 {
                for i in 0..n {
                    betas[[i, comp]] = beta_col[i] / beta_norm;
                }
            }
        }

        Ok(KCCAModel {
            n_components: k,
            alpha: alphas,
            beta: betas,
            correlations,
            x_train: x.to_vec(),
            y_train: y.to_vec(),
            kernel_x: self.kernel_x.clone(),
            kernel_y: self.kernel_y.clone(),
        })
    }
}

impl KCCAModel {
    /// Return canonical correlations.
    pub fn canonical_correlations(&self) -> &[f64] {
        &self.correlations
    }

    /// Project new X samples.
    pub fn transform_x(&self, x_new: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n_new = x_new.len();
        let n_train = self.x_train.len();
        let k = self.n_components;

        // Compute kernel matrix K_new (n_new × n_train)
        let mut kn = Array2::<f64>::zeros((n_new, n_train));
        for (i, xi) in x_new.iter().enumerate() {
            for (j, xj) in self.x_train.iter().enumerate() {
                kn[[i, j]] = self.kernel_x.compute(xi, xj);
            }
        }

        // Project: z_i = K_new_i · alpha_k
        let mut out = vec![vec![0.0f64; k]; n_new];
        for i in 0..n_new {
            for j in 0..k {
                let s: f64 = (0..n_train).map(|t| kn[[i, t]] * self.alpha[[t, j]]).sum();
                out[i][j] = s;
            }
        }
        Ok(out)
    }

    /// Project new Y samples.
    pub fn transform_y(&self, y_new: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n_new = y_new.len();
        let n_train = self.y_train.len();
        let k = self.n_components;

        let mut kn = Array2::<f64>::zeros((n_new, n_train));
        for (i, yi) in y_new.iter().enumerate() {
            for (j, yj) in self.y_train.iter().enumerate() {
                kn[[i, j]] = self.kernel_y.compute(yi, yj);
            }
        }

        let mut out = vec![vec![0.0f64; k]; n_new];
        for i in 0..n_new {
            for j in 0..k {
                let s: f64 = (0..n_train).map(|t| kn[[i, t]] * self.beta[[t, j]]).sum();
                out[i][j] = s;
            }
        }
        Ok(out)
    }
}

// ============================================================================
// GCCA — Generalized CCA (SUM-SQ objective, > 2 views)
// ============================================================================

/// Configuration for Generalized Canonical Correlation Analysis.
///
/// Maximizes the sum-of-squared correlations (SUM-SQ / MAXVAR objective):
/// maximize sum_{v1,v2} corr(G, X_v)  where G is the shared representation.
#[derive(Debug, Clone)]
pub struct GCCA {
    /// Number of canonical components.
    pub n_components: usize,
    /// Regularization for view covariances.
    pub reg: f64,
    /// Maximum number of power-iteration steps.
    pub max_iter: usize,
    /// Convergence tolerance for power iteration.
    pub tol: f64,
}

/// Fitted GCCA model.
#[derive(Debug, Clone)]
pub struct GCCAModel {
    /// Number of components.
    pub n_components: usize,
    /// Projection matrices per view: each is (p_v × n_components).
    pub projections: Vec<Array2<f64>>,
    /// Shared representation G: (n × n_components).
    pub g: Array2<f64>,
    /// Mean of each view used during fit.
    pub view_means: Vec<Vec<f64>>,
}

impl Default for GCCA {
    fn default() -> Self {
        GCCA {
            n_components: 2,
            reg: 1e-4,
            max_iter: 200,
            tol: 1e-6,
        }
    }
}

impl GCCA {
    /// Create a new GCCA instance.
    pub fn new(n_components: usize, reg: f64) -> Self {
        GCCA { n_components, reg, ..Default::default() }
    }

    /// Fit GCCA on multiple views.
    ///
    /// Each view is `views[v]`: a slice of n rows, each a slice of p_v features.
    ///
    /// Algorithm (iterative projection onto view subspaces):
    /// 1. Initialize G = random (n × k).
    /// 2. For each view v: W_v = (C_vv + r I)^{-1} X_v^T G / n.
    /// 3. Update G = sum_v X_v W_v (normalized).
    /// 4. Repeat until convergence.
    pub fn fit(&self, views: &[&[Vec<f64>]]) -> Result<GCCAModel> {
        let n_views = views.len();
        if n_views < 2 {
            return Err(TransformError::InvalidInput(
                "GCCA requires at least 2 views".to_string(),
            ));
        }
        let n = views[0].len();
        for (vi, view) in views.iter().enumerate() {
            if view.len() != n {
                return Err(TransformError::InvalidInput(format!(
                    "View {vi} has {} samples but view 0 has {n}",
                    view.len()
                )));
            }
        }

        let k = self.n_components.min(n);
        let view_means: Vec<Vec<f64>> = views.iter().map(|v| col_means(v)).collect();

        // Center each view and build covariance matrices
        let mut view_arrays: Vec<Array2<f64>> = Vec::with_capacity(n_views);
        for (vi, (view, means)) in views.iter().zip(view_means.iter()).enumerate() {
            let p = view[0].len();
            let mut arr = Array2::<f64>::zeros((n, p));
            for (i, row) in view.iter().enumerate() {
                if row.len() != p {
                    return Err(TransformError::InvalidInput(format!(
                        "View {vi} row {i} length mismatch"
                    )));
                }
                for (j, &v) in row.iter().enumerate() {
                    arr[[i, j]] = v - means[j];
                }
            }
            view_arrays.push(arr);
        }

        // Covariance matrices and their regularized inverses
        let mut cov_inv: Vec<Array2<f64>> = Vec::with_capacity(n_views);
        for arr in view_arrays.iter() {
            let p = arr.ncols();
            // C = X^T X / (n-1)
            let mut c = Array2::<f64>::zeros((p, p));
            for i in 0..n {
                for a in 0..p {
                    for b in 0..p {
                        c[[a, b]] += arr[[i, a]] * arr[[i, b]];
                    }
                }
            }
            let scale = 1.0 / (n as f64 - 1.0).max(1.0);
            c.mapv_inplace(|v| v * scale);
            add_ridge(&mut c, self.reg);

            let ci = inv(&c.view(), None)
                .map_err(|e| TransformError::ComputationError(format!("Cov inv failed: {e}")))?;
            cov_inv.push(ci);
        }

        // Initialize G via SVD of the concatenated (whitened) views
        let mut g = Array2::<f64>::zeros((n, k));
        // Simple init: use first view's first k left singular vectors
        {
            let first = &view_arrays[0];
            let (u, _s, _vt) = svd(&first.view(), true, None).map_err(|e| {
                TransformError::ComputationError(format!("GCCA init SVD failed: {e}"))
            })?;
            let cols = k.min(u.ncols());
            for i in 0..n {
                for j in 0..cols {
                    g[[i, j]] = u[[i, j]];
                }
            }
        }

        // Power iteration
        for _iter in 0..self.max_iter {
            let g_old = g.clone();

            // W_v = C_vv^{-1} X_v^T G / n  →  projection for view v
            // New G = sum_v X_v W_v  (unnormalized)
            let mut g_new = Array2::<f64>::zeros((n, k));
            for (arr, ci) in view_arrays.iter().zip(cov_inv.iter()) {
                // xg = X_v^T G: (p_v × k)
                let xg = arr.t().dot(&g);
                // w = C_vv^{-1} xg / n: (p_v × k)
                let w = ci.dot(&xg) / (n as f64);
                // contrib = X_v w: (n × k)
                let contrib = arr.dot(&w);
                g_new = g_new + contrib;
            }

            // Orthonormalize G via QR
            let (u, _s, _vt) = svd(&g_new.view(), true, None).map_err(|e| {
                TransformError::ComputationError(format!("GCCA power iter SVD: {e}"))
            })?;
            let cols = k.min(u.ncols());
            g = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                for j in 0..cols {
                    g[[i, j]] = u[[i, j]];
                }
            }

            // Convergence check: Frobenius norm of change
            let diff: f64 = g.iter().zip(g_old.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            if diff < self.tol {
                break;
            }
        }

        // Final projections W_v = C_vv^{-1} X_v^T G / n
        let mut projections: Vec<Array2<f64>> = Vec::with_capacity(n_views);
        for (arr, ci) in view_arrays.iter().zip(cov_inv.iter()) {
            let xg = arr.t().dot(&g);
            let w = ci.dot(&xg) / (n as f64);
            projections.push(w);
        }

        Ok(GCCAModel {
            n_components: k,
            projections,
            g,
            view_means,
        })
    }
}

impl GCCAModel {
    /// Transform a new view using the learned projection.
    ///
    /// `view_idx` is the 0-based index of the view (must match fitting order).
    pub fn transform_view(&self, view: &[Vec<f64>], view_idx: usize) -> Result<Vec<Vec<f64>>> {
        if view_idx >= self.projections.len() {
            return Err(TransformError::InvalidInput(format!(
                "view_idx {view_idx} out of range (have {} views)",
                self.projections.len()
            )));
        }
        let w = &self.projections[view_idx];
        let means = &self.view_means[view_idx];
        let p = w.nrows();
        let k = self.n_components;
        let n = view.len();

        let mut out = vec![vec![0.0f64; k]; n];
        for (i, row) in view.iter().enumerate() {
            if row.len() != p {
                return Err(TransformError::InvalidInput(format!(
                    "Row {i}: expected {p} features, got {}",
                    row.len()
                )));
            }
            let centered: Vec<f64> = row.iter().zip(means.iter()).map(|(v, m)| v - m).collect();
            for j in 0..k {
                let s: f64 = (0..p).map(|fi| centered[fi] * w[[fi, j]]).sum();
                out[i][j] = s;
            }
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

    fn make_correlated_views(n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // X and Y share a latent signal
        let x: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.1;
                vec![t.sin(), t.cos(), t * 0.01]
            })
            .collect();
        let y: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.1;
                vec![t.sin() + 0.05 * (i as f64).sin(), t.cos() * 1.1]
            })
            .collect();
        (x, y)
    }

    #[test]
    fn test_cca_basic() {
        let (x, y) = make_correlated_views(80);
        let cca = CCA::new(2, 1e-4);
        let model = cca.fit(&x, &y).expect("CCA fit failed");

        assert_eq!(model.n_components, 2);
        assert_eq!(model.correlations.len(), 2);

        // Correlations should be in [0, 1]
        for &c in &model.correlations {
            assert!(c >= 0.0 && c <= 1.0 + 1e-9, "correlation {c} out of [0,1]");
        }

        let zx = model.transform_x(&x).expect("transform_x failed");
        let zy = model.transform_y(&y).expect("transform_y failed");
        assert_eq!(zx.len(), 80);
        assert_eq!(zy.len(), 80);
        assert_eq!(zx[0].len(), 2);
    }

    #[test]
    fn test_cca_correlations_ordered() {
        let (x, y) = make_correlated_views(100);
        let cca = CCA::new(2, 1e-4);
        let model = cca.fit(&x, &y).expect("CCA fit");
        let corrs = model.canonical_correlations();
        assert!(corrs[0] >= corrs[1] - 1e-9, "correlations should be non-increasing");
    }

    #[test]
    fn test_kcca_basic() {
        let (x, y) = make_correlated_views(40);
        let kcca = KCCA::new(
            2,
            KernelType::RBF { gamma: 0.5 },
            KernelType::RBF { gamma: 0.5 },
            1e-2,
        );
        let model = kcca.fit(&x, &y).expect("KCCA fit failed");
        assert_eq!(model.n_components, 2);

        let zx = model.transform_x(&x).expect("KCCA transform_x");
        assert_eq!(zx.len(), 40);
        assert_eq!(zx[0].len(), 2);
    }

    #[test]
    fn test_gcca_basic() {
        let (x, y) = make_correlated_views(60);
        let z: Vec<Vec<f64>> = x.iter().zip(y.iter()).map(|(xi, yi)| {
            let mut v = xi.clone();
            v.extend_from_slice(yi);
            v
        }).collect();

        let gcca = GCCA::new(2, 1e-3);
        let views: &[&[Vec<f64>]] = &[x.as_slice(), y.as_slice(), z.as_slice()];
        let model = gcca.fit(views).expect("GCCA fit failed");
        assert_eq!(model.n_components, 2);
        assert_eq!(model.projections.len(), 3);

        let embed = model.transform_view(&x, 0).expect("transform_view");
        assert_eq!(embed.len(), 60);
        assert_eq!(embed[0].len(), 2);
    }

    #[test]
    fn test_cca_error_mismatched_rows() {
        let x = vec![vec![1.0, 2.0]; 10];
        let y = vec![vec![1.0]; 5];
        let cca = CCA::default();
        assert!(cca.fit(&x, &y).is_err());
    }
}
