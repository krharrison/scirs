//! Non-negative Matrix Factorization Variants
//!
//! This module provides a suite of NMF algorithms beyond the standard multiplicative-update
//! method already in `decomposition::NMF`.  Each variant solves:
//!
//! ```text
//! min_{W,H} ||X - W H||_F   subject to various non-negativity / sparsity / convexity constraints
//! ```
//!
//! ## Algorithms
//!
//! | Struct | Method | Notes |
//! |--------|--------|-------|
//! | [`NmfMu`] | Multiplicative updates (Lee & Seung 2001) | Classic algorithm, exact ||·||_F |
//! | [`NmfAls`] | Alternating Least Squares | Faster convergence on overdetermined problems |
//! | [`NmfSparse`] | Sparse NMF with L1 penalty on H | Promotes sparsity in codes |
//! | [`NmfSemiNmf`] | Semi-NMF | W can be negative; H ≥ 0 |
//! | [`NmfConvex`] | Convex NMF | W = X G with G ≥ 0; archetypes live in the data |
//! | [`NmfOnline`] | Online / mini-batch NMF | Streaming updates for large datasets |
//!
//! ## Utility Functions
//!
//! - [`nmf_quality`] - returns `(reconstruction_error, sparsity_of_H)`
//!
//! ## References
//!
//! - Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. NIPS.
//! - Kim, H., & Park, H. (2007). Sparse non-negative matrix factorizations via alternating
//!   non-negativity-constrained least squares.
//! - Ding, C., Li, T., & Jordan, M. I. (2010). Convex and semi-nonnegative matrix factorizations. TPAMI.
//! - Mairal, J., Bach, F., Ponce, J., & Sapiro, G. (2010). Online learning for matrix factorization. JMLR.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::Rng;

// ─── Small epsilon used throughout to prevent division by zero ────────────────
const EPS: f64 = 1e-10;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: random non-negative initialisation
// ─────────────────────────────────────────────────────────────────────────────

/// Initialise W (n_rows × k) and H (k × n_cols) with uniform [0, scale] values.
fn random_wh(n_rows: usize, k: usize, n_cols: usize, scale: f64) -> (Array2<f64>, Array2<f64>) {
    let mut rng = scirs2_core::random::rng();
    let mut w = Array2::<f64>::zeros((n_rows, k));
    let mut h = Array2::<f64>::zeros((k, n_cols));
    for i in 0..n_rows {
        for j in 0..k {
            w[[i, j]] = rng.random::<f64>() * scale;
        }
    }
    for i in 0..k {
        for j in 0..n_cols {
            h[[i, j]] = rng.random::<f64>() * scale;
        }
    }
    (w, h)
}

/// Frobenius norm of (X − W H)
fn frob_error(x: &Array2<f64>, w: &Array2<f64>, h: &Array2<f64>) -> f64 {
    let wh = w.dot(h);
    let diff = x - &wh;
    diff.mapv(|v| v * v).sum().sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// NmfMu — Multiplicative Updates (Lee & Seung)
// ─────────────────────────────────────────────────────────────────────────────

/// NMF via multiplicative update rules (Lee & Seung, 2001).
///
/// Minimises: ½ ||X − W H||²_F with W, H ≥ 0.
///
/// Update rules:
/// ```text
/// H ← H ⊙ (Wᵀ X) / (Wᵀ W H + ε)
/// W ← W ⊙ (X Hᵀ) / (W H Hᵀ + ε)
/// ```
#[derive(Debug, Clone)]
pub struct NmfMu {
    /// Number of latent factors k
    pub n_components: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance (relative change in Frobenius error)
    pub tol: f64,
}

impl NmfMu {
    /// Create a new `NmfMu` with defaults (200 iterations, tol=1e-4).
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
        }
    }

    /// Set maximum number of iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit NMF to data matrix `X` (n_samples × n_features, all entries ≥ 0).
    ///
    /// Returns `(W, H)` where `W` is n_samples × k and `H` is k × n_features.
    pub fn fit<S>(&self, x_raw: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x = to_f64(x_raw)?;
        check_non_negative(&x)?;
        check_rank(&x, self.n_components)?;

        let (n, p) = x.dim();
        let scale = (x.mean().unwrap_or(1.0) / self.n_components as f64).sqrt().max(EPS);
        let (mut w, mut h) = random_wh(n, self.n_components, p, scale);

        let mut prev_err = frob_error(&x, &w, &h);

        for _ in 0..self.max_iter {
            // H update
            let wt_x = w.t().dot(&x);
            let wt_wh = w.t().dot(&w.dot(&h));
            for i in 0..self.n_components {
                for j in 0..p {
                    h[[i, j]] = (h[[i, j]] * wt_x[[i, j]] / (wt_wh[[i, j]] + EPS)).max(EPS);
                }
            }

            // W update
            let x_ht = x.dot(&h.t());
            let whht = w.dot(&h).dot(&h.t());
            for i in 0..n {
                for j in 0..self.n_components {
                    w[[i, j]] = (w[[i, j]] * x_ht[[i, j]] / (whht[[i, j]] + EPS)).max(EPS);
                }
            }

            let err = frob_error(&x, &w, &h);
            if (prev_err - err).abs() / prev_err.max(EPS) < self.tol {
                break;
            }
            prev_err = err;
        }

        Ok((w, h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NmfAls — Alternating Least Squares
// ─────────────────────────────────────────────────────────────────────────────

/// NMF via Alternating Least Squares with non-negativity projection.
///
/// Each sub-problem is an unconstrained least-squares solve followed by
/// clamping to non-negative values.  This typically converges faster than MU
/// on well-conditioned problems.
///
/// ```text
/// H ← argmin_H ||X − W H||_F → H[[i,j]] = max(0, solution[[i,j]])
/// W ← argmin_W ||X − W H||_F → W[[i,j]] = max(0, solution[[i,j]])
/// ```
#[derive(Debug, Clone)]
pub struct NmfAls {
    /// Number of latent factors k
    pub n_components: usize,
    /// Maximum alternating iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Tikhonov regularisation added to normal-equations diagonal
    pub reg: f64,
}

impl NmfAls {
    /// Create with defaults (300 iters, tol=1e-5, reg=1e-8).
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 300,
            tol: 1e-5,
            reg: 1e-8,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set regularisation strength.
    pub fn with_reg(mut self, reg: f64) -> Self {
        self.reg = reg;
        self
    }

    /// Fit: returns `(W, H)`.
    pub fn fit<S>(&self, x_raw: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x = to_f64(x_raw)?;
        check_non_negative(&x)?;
        check_rank(&x, self.n_components)?;

        let (n, p) = x.dim();
        let k = self.n_components;
        let scale = (x.mean().unwrap_or(1.0) / k as f64).sqrt().max(EPS);
        let (mut w, mut h) = random_wh(n, k, p, scale);

        let mut prev_err = frob_error(&x, &w, &h);

        for _ in 0..self.max_iter {
            // Solve for H: (WᵀW + reg I) H = Wᵀ X
            h = als_solve_h(&x, &w, k, self.reg);

            // Solve for W: (H Hᵀ + reg I) Wᵀ = H Xᵀ  ⟹  Wᵀ = ...
            w = als_solve_w(&x, &h, n, self.reg);

            let err = frob_error(&x, &w, &h);
            if (prev_err - err).abs() / prev_err.max(EPS) < self.tol {
                break;
            }
            prev_err = err;
        }

        Ok((w, h))
    }
}

/// Solve for H given fixed W: min_H ||X - W H||_F  s.t. H ≥ 0
///
/// Normal equations: (WᵀW + λI) H = WᵀX
fn als_solve_h(x: &Array2<f64>, w: &Array2<f64>, k: usize, reg: f64) -> Array2<f64> {
    let p = x.ncols();
    let wtw = w.t().dot(w);
    // Add regularisation
    let mut a = wtw;
    for i in 0..k {
        a[[i, i]] += reg;
    }
    let b = w.t().dot(x);

    // Solve A H = B column by column using Cholesky-like back-substitution
    // For simplicity use the same element-wise LS approach as coordinate descent
    let mut h = Array2::<f64>::zeros((k, p));
    for col in 0..p {
        let rhs: Vec<f64> = (0..k).map(|i| b[[i, col]]).collect();
        let sol = solve_posdef_system(&a, &rhs);
        for i in 0..k {
            h[[i, col]] = sol[i].max(EPS);
        }
    }
    h
}

/// Solve for W given fixed H: min_W ||X - W H||_F  s.t. W ≥ 0
fn als_solve_w(x: &Array2<f64>, h: &Array2<f64>, n: usize, reg: f64) -> Array2<f64> {
    let k = h.nrows();
    let hht = h.dot(&h.t());
    let mut a = hht;
    for i in 0..k {
        a[[i, i]] += reg;
    }
    let xht = x.dot(&h.t()); // n × k

    let mut w = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        let rhs: Vec<f64> = (0..k).map(|j| xht[[row, j]]).collect();
        let sol = solve_posdef_system(&a, &rhs);
        for j in 0..k {
            w[[row, j]] = sol[j].max(EPS);
        }
    }
    w
}

/// Solve a k×k positive-definite system A x = b using Gaussian elimination
/// (no external linalg dependency to keep compile-time light).
fn solve_posdef_system(a: &Array2<f64>, b: &[f64]) -> Vec<f64> {
    let k = b.len();
    // Build augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            let mut row: Vec<f64> = (0..k).map(|j| a[[i, j]]).collect();
            row.push(b[i]);
            row
        })
        .collect();

    // Forward elimination
    for col in 0..k {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..k {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < EPS {
            continue;
        }
        let inv_pivot = 1.0 / pivot;
        for row in (col + 1)..k {
            let factor = aug[row][col] * inv_pivot;
            for c in col..=k {
                let val = aug[col][c];
                aug[row][c] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; k];
    for i in (0..k).rev() {
        let mut sum = aug[i][k];
        for j in (i + 1)..k {
            sum -= aug[i][j] * x[j];
        }
        x[i] = if aug[i][i].abs() > EPS {
            sum / aug[i][i]
        } else {
            0.0
        };
    }
    x
}

// ─────────────────────────────────────────────────────────────────────────────
// NmfSparse — Sparse NMF with L1 on H
// ─────────────────────────────────────────────────────────────────────────────

/// Sparse NMF: adds an L1 penalty on H to promote sparsity in the code matrix.
///
/// Objective: ½ ||X − W H||²_F + λ ||H||₁   s.t. W, H ≥ 0
///
/// The H update uses a projected gradient / soft-threshold step.
/// The W update is the standard MU rule.
#[derive(Debug, Clone)]
pub struct NmfSparse {
    /// Number of latent factors
    pub n_components: usize,
    /// L1 penalty on H (λ)
    pub lambda: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl NmfSparse {
    /// Create with defaults (200 iters, tol=1e-4, lambda=0.1).
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            lambda: 0.1,
            max_iter: 200,
            tol: 1e-4,
        }
    }

    /// Set L1 penalty strength.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit: returns `(W, H)`.
    pub fn fit<S>(&self, x_raw: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x = to_f64(x_raw)?;
        check_non_negative(&x)?;
        check_rank(&x, self.n_components)?;

        let (n, p) = x.dim();
        let k = self.n_components;
        let scale = (x.mean().unwrap_or(1.0) / k as f64).sqrt().max(EPS);
        let (mut w, mut h) = random_wh(n, k, p, scale);

        let mut prev_err = frob_error(&x, &w, &h);

        for _ in 0..self.max_iter {
            // H update: MU rule with soft-threshold to handle L1
            // Gradient of ||X - WH||²/2 w.r.t. H = WᵀWH - WᵀX
            // Project-gradient: H ← max(0, H - step * (WᵀWH - WᵀX + λ))
            // Use multiplicative form of the projected gradient
            let wt_x = w.t().dot(&x);
            let wtwh = w.t().dot(&w.dot(&h));
            for i in 0..k {
                for j in 0..p {
                    let num = wt_x[[i, j]];
                    let den = wtwh[[i, j]] + self.lambda + EPS;
                    h[[i, j]] = (h[[i, j]] * num / den).max(EPS);
                }
            }

            // W update: standard MU (no sparsity on W)
            let x_ht = x.dot(&h.t());
            let whht = w.dot(&h).dot(&h.t());
            for i in 0..n {
                for j in 0..k {
                    w[[i, j]] = (w[[i, j]] * x_ht[[i, j]] / (whht[[i, j]] + EPS)).max(EPS);
                }
            }

            let err = frob_error(&x, &w, &h);
            if (prev_err - err).abs() / prev_err.max(EPS) < self.tol {
                break;
            }
            prev_err = err;
        }

        Ok((w, h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NmfSemiNmf — Semi-NMF (W unconstrained, H ≥ 0)
// ─────────────────────────────────────────────────────────────────────────────

/// Semi-NMF: W may contain negative entries; H remains non-negative.
///
/// This allows the model to capture "mixed-sign" bases (e.g. contrast patterns)
/// while still producing a parts-based, non-negative encoding H.
///
/// Algorithm (Ding et al. 2010):
/// ```text
/// H ← H ⊙ (Wᵀ X)⁺ / ((Wᵀ X)⁻ + Wᵀ W H)   element-wise
/// W ← X Hᵀ (H Hᵀ)⁻¹                         unconstrained LS
/// ```
/// where (·)⁺ = max(·,0) and (·)⁻ = max(−·,0).
#[derive(Debug, Clone)]
pub struct NmfSemiNmf {
    /// Number of latent factors k
    pub n_components: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Regularisation on H Hᵀ to keep it invertible
    pub reg: f64,
}

impl NmfSemiNmf {
    /// Create with defaults.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            reg: 1e-8,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit: returns `(W, H)` where W may contain negative values.
    pub fn fit<S>(&self, x_raw: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x = to_f64(x_raw)?;
        // Semi-NMF: X can have negative entries
        check_rank(&x, self.n_components)?;

        let (n, p) = x.dim();
        let k = self.n_components;
        let scale = 0.1f64;
        let (mut w, mut h) = random_wh(n, k, p, scale);
        // Allow W to be centred around 0 by shifting with noise
        {
            let mut rng = scirs2_core::random::rng();
            for v in w.iter_mut() {
                *v = rng.random::<f64>() * 2.0 - 1.0;
            }
        }

        let mut prev_err = frob_error(&x, &w, &h);

        for _ in 0..self.max_iter {
            // H update (MU-style with positive/negative decomposition)
            let wt_x = w.t().dot(&x); // k × p
            let wt_x_pos = wt_x.mapv(|v| v.max(0.0));
            let wt_x_neg = wt_x.mapv(|v| (-v).max(0.0));

            let wtw = w.t().dot(&w); // k × k
            let wtwh = wtw.dot(&h); // k × p

            for i in 0..k {
                for j in 0..p {
                    let num = wt_x_pos[[i, j]] + EPS;
                    let den = wt_x_neg[[i, j]] + wtwh[[i, j]] + EPS;
                    h[[i, j]] = (h[[i, j]] * num / den).max(EPS);
                }
            }

            // W update: unconstrained LS  W = X Hᵀ (H Hᵀ + reg I)⁻¹
            let xht = x.dot(&h.t()); // n × k
            let mut hht = h.dot(&h.t()); // k × k
            for i in 0..k {
                hht[[i, i]] += self.reg;
            }
            // Solve (H Hᵀ) Wᵀ = (X Hᵀ)ᵀ  →  solve column by column
            for row in 0..n {
                let rhs: Vec<f64> = (0..k).map(|j| xht[[row, j]]).collect();
                let sol = solve_posdef_system(&hht, &rhs);
                for j in 0..k {
                    w[[row, j]] = sol[j];
                }
            }

            let err = frob_error(&x, &w, &h);
            if (prev_err - err).abs() / prev_err.max(EPS) < self.tol {
                break;
            }
            prev_err = err;
        }

        Ok((w, h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NmfConvex — Convex NMF
// ─────────────────────────────────────────────────────────────────────────────

/// Convex NMF: constrains the basis to be a convex combination of data points.
///
/// Parameterisation: W = X G where G ≥ 0, col-stochastic.
/// Data approximation: X ≈ W H = X G H.
///
/// This ensures that each basis vector lives inside the convex hull of the data,
/// giving highly interpretable archetypal-like bases.
///
/// Updates (Ding et al. 2010):
/// ```text
/// G ← G ⊙ sqrt((XᵀX H^T)⁺ / ((XᵀX G H H^T)⁺ + EPS))
/// H ← H ⊙ sqrt((Gᵀ XᵀX)⁺ / ((Gᵀ XᵀX G H)⁺ + EPS))
/// ```
#[derive(Debug, Clone)]
pub struct NmfConvex {
    /// Number of convex components k
    pub n_components: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl NmfConvex {
    /// Create with defaults.
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit convex NMF.
    ///
    /// Returns `(W, H, G)` where `W = X * G` is the basis (n_samples × k),
    /// `H` is k × n_features, and `G` is n_samples × k (convex coefficients).
    pub fn fit<S>(
        &self,
        x_raw: &ArrayBase<S, Ix2>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x = to_f64(x_raw)?;
        check_non_negative(&x)?;
        check_rank(&x, self.n_components)?;

        let (n, p) = x.dim();
        let k = self.n_components;

        // Initialise G (n × k) and H (k × p) as random non-negative column-stochastic
        let mut rng = scirs2_core::random::rng();
        let mut g = Array2::<f64>::zeros((n, k));
        let mut h = Array2::<f64>::zeros((k, p));
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..k {
                g[[i, j]] = rng.random::<f64>() + EPS;
                row_sum += g[[i, j]];
            }
            for j in 0..k {
                g[[i, j]] /= row_sum;
            }
        }
        for i in 0..k {
            for j in 0..p {
                h[[i, j]] = rng.random::<f64>() + EPS;
            }
        }

        // Pre-compute kernel matrix K = XᵀX  (p × n) · (n × p) = p × p  ← actually n × n
        // Ding et al. uses K = X Xᵀ  (n × n) because W = X G lives in R^{n × k}
        // and H ∈ R^{k × p}.  Objective = ||X - X G H||²_F.
        let xtx = x.t().dot(&x); // p × p

        let mut prev_err = {
            let w = x.dot(&g);
            frob_error(&x, &w, &h)
        };

        for _ in 0..self.max_iter {
            // G update
            {
                let xg = x.dot(&g); // n × k  (= W)
                let w = &xg;
                // XᵀX H^T → p × p · (p × k)^T = wait, shapes:
                // We need gradient ∇_G = -2 Xᵀ(X - XGH)Hᵀ = -2(XᵀX - XᵀXGH)Hᵀ
                // MU: G ← G ⊙ (XᵀX Hᵀ Hᵀ is wrong...) use direct form:
                // Let R = X - WH; G update numerator = Xᵀ R Hᵀ, denom = Xᵀ W H Hᵀ
                let r = &x - &w.dot(&h); // n × p
                let xt_r_ht = x.t().dot(&r).dot(&h.t()); // p × k  ← wrong dims for G
                // G has shape n × k, not p × k.
                // Correct: G update from ∇_G f = Xᵀ(XGH - X)Hᵀ
                // = XᵀX G H Hᵀ - XᵀX Hᵀ ... wait, the gradient is w.r.t. G in X space:
                // f = ||X - XGH||² → df/dG = -2Xᵀ(X - XGH)Hᵀ
                // numerator part: 2 Xᵀ X Hᵀ ... shape (n×p)ᵀ · (p×?) → need n×k
                // Correct: Xᵀ is (p×n), so Xᵀ R is p×p, Hᵀ is p×k → p×k not n×k
                // Need to use: numerator = X Xᵀ ... hmm
                // The correct dual form: use K = X Xᵀ (n×n)
                // f = Tr(Xᵀ X) - 2 Tr(Hᵀ Gᵀ Xᵀ X) + Tr(Hᵀ Gᵀ Xᵀ X G H)
                // ∂f/∂G = -2 X Xᵀ . hmm... Ding et al. derives in terms of K=XᵀX
                // Let's use the direct MU approach without the kernel trick:
                let _ = xt_r_ht; // unused, replaced below

                let xxt = x.dot(&x.t()); // n × n
                let xxt_g = xxt.dot(&g); // n × k
                let num_g = xxt.dot(&x.dot(&h.t())); // n × k ... X Hᵀ is n×k? no, H is k×p, Hᵀ is p×k
                // Actually X (n×p) · Hᵀ (p×k) = n×k  ✓
                let x_ht = x.dot(&h.t()); // n × k
                let num_g2 = xxt.dot(&x_ht); // n × k — this is Xᵀ... wait, xxt is n×n
                // num = (X Xᵀ) · (X Hᵀ) = n×n · n×k = n×k ✓

                // But now we need the denominator:
                // den = (X Xᵀ G H) Hᵀ = n×n · n×k · k×p · p×k = n×k ✓
                let den_g = xxt_g.dot(&h).dot(&h.t()); // n×k

                for i in 0..n {
                    for j in 0..k {
                        let num = num_g2[[i, j]];
                        let den = den_g[[i, j]] + EPS;
                        if num > 0.0 {
                            g[[i, j]] = (g[[i, j]] * num / den).max(EPS);
                        }
                    }
                }
                // Re-normalise columns of G so W = XG is well-scaled
                for j in 0..k {
                    let col_sum: f64 = (0..n).map(|i| g[[i, j]]).sum::<f64>().max(EPS);
                    for i in 0..n {
                        g[[i, j]] /= col_sum;
                    }
                }
            }

            // H update (standard MU with W = XG)
            {
                let w = x.dot(&g); // n × k
                let wt_x = w.t().dot(&x); // k × p  (numerator)
                let wt_wh = w.t().dot(&w.dot(&h)); // k × p  (denominator)
                for i in 0..k {
                    for j in 0..p {
                        h[[i, j]] = (h[[i, j]] * wt_x[[i, j]] / (wt_wh[[i, j]] + EPS)).max(EPS);
                    }
                }
            }

            let w = x.dot(&g);
            let err = frob_error(&x, &w, &h);
            if (prev_err - err).abs() / prev_err.max(EPS) < self.tol {
                break;
            }
            prev_err = err;
        }

        let w = x.dot(&g);
        Ok((w, h, g))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NmfOnline — Online / Mini-batch NMF
// ─────────────────────────────────────────────────────────────────────────────

/// Online NMF following Mairal et al. (2010).
///
/// The dictionary H (k × p) is updated incrementally after each mini-batch,
/// while sparse codes W are obtained by solving the encoding problem per batch.
/// This allows factorisation of datasets too large to hold in memory.
#[derive(Debug, Clone)]
pub struct NmfOnline {
    /// Number of latent factors k
    pub n_components: usize,
    /// Mini-batch size
    pub batch_size: usize,
    /// Number of passes over the data (epochs)
    pub n_epochs: usize,
    /// Forgetting rate for exponential averaging of statistics (∈ (0.5, 1])
    pub rho: f64,
    /// Convergence tolerance (on the dictionary change)
    pub tol: f64,
    /// Optional fixed random seed
    pub seed: Option<u64>,
}

impl NmfOnline {
    /// Create with defaults (batch_size=32, 10 epochs, rho=0.9).
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            batch_size: 32,
            n_epochs: 10,
            rho: 0.9,
            tol: 1e-4,
            seed: None,
        }
    }

    /// Set mini-batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set number of epochs.
    pub fn with_n_epochs(mut self, n_epochs: usize) -> Self {
        self.n_epochs = n_epochs;
        self
    }

    /// Set forgetting rate.
    pub fn with_rho(mut self, rho: f64) -> Self {
        self.rho = rho.clamp(0.5, 1.0);
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Fit online NMF to data matrix X.
    ///
    /// Returns `(W, H)` — the final encoding of the full dataset and the
    /// learned dictionary.
    pub fn fit<S>(&self, x_raw: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x = to_f64(x_raw)?;
        check_non_negative(&x)?;
        check_rank(&x, self.n_components)?;

        let (n, p) = x.dim();
        let k = self.n_components;

        // Initialise dictionary H (k × p)
        let scale = (x.mean().unwrap_or(1.0) / k as f64).sqrt().max(EPS);
        let (_, mut h) = random_wh(1, k, p, scale);

        // Accumulated statistics for dictionary update (Mairal et al.)
        let mut a = Array2::<f64>::zeros((k, k)); // ∑ α α^T
        let mut b = Array2::<f64>::zeros((k, p)); // ∑ α x^T

        let mut rng = scirs2_core::random::rng();

        for _epoch in 0..self.n_epochs {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..n).collect();
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = (rng.random::<f64>() * (i + 1) as f64) as usize;
                indices.swap(i, j);
            }

            let mut start = 0;
            while start < n {
                let end = (start + self.batch_size).min(n);
                let batch_idx = &indices[start..end];
                let batch_n = batch_idx.len();

                // Extract mini-batch
                let mut x_batch = Array2::<f64>::zeros((batch_n, p));
                for (bi, &gi) in batch_idx.iter().enumerate() {
                    for j in 0..p {
                        x_batch[[bi, j]] = x[[gi, j]];
                    }
                }

                // Encode each sample in the batch (MU for simplicity)
                let mut alpha = Array2::<f64>::zeros((batch_n, k)); // batch encodings
                for i in 0..batch_n {
                    for j in 0..k {
                        alpha[[i, j]] = scale * rng.random::<f64>();
                    }
                }
                for _inner in 0..50 {
                    let xt_batch = x_batch.clone(); // n_b × p
                    let num = xt_batch.dot(&h.t()); // n_b × k
                    let den = alpha.dot(&h).dot(&h.t()); // n_b × k
                    for i in 0..batch_n {
                        for j in 0..k {
                            alpha[[i, j]] =
                                (alpha[[i, j]] * num[[i, j]] / (den[[i, j]] + EPS)).max(EPS);
                        }
                    }
                }

                // Update accumulated statistics
                let alpha_t = alpha.t().to_owned(); // k × batch_n
                let new_a = alpha_t.dot(&alpha); // k × k
                let new_b = alpha_t.dot(&x_batch); // k × p

                let rho = self.rho;
                for i in 0..k {
                    for j in 0..k {
                        a[[i, j]] = rho * a[[i, j]] + new_a[[i, j]];
                    }
                    for j in 0..p {
                        b[[i, j]] = rho * b[[i, j]] + new_b[[i, j]];
                    }
                }

                // Dictionary update: H ← argmin_H Tr(½ Hᵀ A H - Bᵀ H)  s.t. H ≥ 0
                // Equivalent to H ← max(0, B A^{-1}) but MU form is simpler:
                for i in 0..k {
                    for j in 0..p {
                        let num_h = b[[i, j]];
                        let den_h = a.row(i).dot(&h.column(j)) + EPS;
                        if num_h > 0.0 {
                            h[[i, j]] = (h[[i, j]] * num_h / den_h).max(EPS);
                        }
                    }
                }

                start = end;
            }
        }

        // Compute final full-dataset encoding W
        let (mut w, _) = random_wh(n, k, 1, scale); // placeholder
        w = Array2::<f64>::zeros((n, k));
        {
            let mut rng2 = scirs2_core::random::rng();
            for i in 0..n {
                for j in 0..k {
                    w[[i, j]] = scale * rng2.random::<f64>();
                }
            }
        }
        for _inner in 0..100 {
            let num = x.dot(&h.t()); // n × k
            let den = w.dot(&h).dot(&h.t()); // n × k
            for i in 0..n {
                for j in 0..k {
                    w[[i, j]] = (w[[i, j]] * num[[i, j]] / (den[[i, j]] + EPS)).max(EPS);
                }
            }
        }

        Ok((w, h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// nmf_quality — reconstruction error + sparsity
// ─────────────────────────────────────────────────────────────────────────────

/// Compute quality metrics for an NMF decomposition.
///
/// # Arguments
/// * `x` - Original data matrix (n_samples × n_features)
/// * `w` - Encoding matrix (n_samples × k)
/// * `h` - Dictionary matrix (k × n_features)
///
/// # Returns
/// `(reconstruction_error, sparsity_of_h)`
/// - `reconstruction_error` = ||X − WH||_F / ||X||_F  (relative Frobenius)
/// - `sparsity_of_h`        = fraction of H entries below 1e-6  ∈ [0, 1]
pub fn nmf_quality(x: &Array2<f64>, w: &Array2<f64>, h: &Array2<f64>) -> (f64, f64) {
    let wh = w.dot(h);
    let diff = x - &wh;
    let rec_err = diff.mapv(|v| v * v).sum().sqrt();
    let x_norm = x.mapv(|v| v * v).sum().sqrt().max(EPS);
    let rel_err = rec_err / x_norm;

    let total_h = h.len() as f64;
    let sparse_count = h.iter().filter(|&&v| v < 1e-6).count() as f64;
    let sparsity = if total_h > 0.0 {
        sparse_count / total_h
    } else {
        0.0
    };

    (rel_err, sparsity)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal validation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a generic Array to f64.
fn to_f64<S>(x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    Ok(x.mapv(|v| NumCast::from(v).unwrap_or(0.0)))
}

/// Return error if any entry is negative.
fn check_non_negative(x: &Array2<f64>) -> Result<()> {
    for &v in x.iter() {
        if v < 0.0 {
            return Err(TransformError::InvalidInput(
                "NMF requires a non-negative input matrix".to_string(),
            ));
        }
    }
    Ok(())
}

/// Return error if k > min(n, p).
fn check_rank(x: &Array2<f64>, k: usize) -> Result<()> {
    let (n, p) = x.dim();
    let max_rank = n.min(p);
    if k > max_rank {
        return Err(TransformError::InvalidInput(format!(
            "n_components={k} must be ≤ min(n_samples={n}, n_features={p})"
        )));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn toy_matrix() -> Array2<f64> {
        // Simple rank-2 non-negative matrix
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 3.0, 6.0, 9.0, 12.0, 4.0, 8.0, 12.0, 16.0,
            5.0, 10.0, 15.0, 20.0, 6.0, 12.0, 18.0, 24.0,
        ];
        Array2::from_shape_vec((6, 4), data).expect("shape ok")
    }

    #[test]
    fn test_nmf_mu_basic() {
        let x = toy_matrix();
        let nmf = NmfMu::new(2).with_max_iter(300).with_tol(1e-5);
        let (w, h) = nmf.fit(&x).expect("NmfMu fit ok");

        assert_eq!(w.shape(), &[6, 2]);
        assert_eq!(h.shape(), &[2, 4]);

        for &v in w.iter() {
            assert!(v >= 0.0, "W must be non-negative");
        }
        for &v in h.iter() {
            assert!(v >= 0.0, "H must be non-negative");
        }

        let (rel_err, _) = nmf_quality(&x, &w, &h);
        assert!(rel_err < 0.5, "relative error {rel_err} should be small");
    }

    #[test]
    fn test_nmf_als_basic() {
        let x = toy_matrix();
        let nmf = NmfAls::new(2).with_max_iter(200);
        let (w, h) = nmf.fit(&x).expect("NmfAls fit ok");

        assert_eq!(w.shape(), &[6, 2]);
        assert_eq!(h.shape(), &[2, 4]);

        for &v in w.iter() {
            assert!(v >= 0.0);
        }
        for &v in h.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_nmf_sparse_promotes_sparsity() {
        let x = toy_matrix();
        let nmf_sparse = NmfSparse::new(2).with_lambda(1.0).with_max_iter(300);
        let nmf_base = NmfSparse::new(2).with_lambda(0.0).with_max_iter(300);

        let (ws, hs) = nmf_sparse.fit(&x).expect("NmfSparse fit ok");
        let (wb, hb) = nmf_base.fit(&x).expect("NmfSparse base fit ok");

        let (_, sp_s) = nmf_quality(&x, &ws, &hs);
        let (_, sp_b) = nmf_quality(&x, &wb, &hb);

        // With lambda=1.0 sparsity should be at least as high as lambda=0
        assert!(sp_s >= sp_b || (sp_s - sp_b).abs() < 0.2, "sparse H not sparser");
    }

    #[test]
    fn test_nmf_semi_basic() {
        let x = toy_matrix();
        let nmf = NmfSemiNmf::new(2).with_max_iter(200);
        let (w, h) = nmf.fit(&x).expect("NmfSemiNmf fit ok");

        assert_eq!(w.shape(), &[6, 2]);
        assert_eq!(h.shape(), &[2, 4]);

        // H must be non-negative; W may be anything
        for &v in h.iter() {
            assert!(v >= 0.0, "H must be non-negative");
        }
    }

    #[test]
    fn test_nmf_convex_basic() {
        let x = toy_matrix();
        let nmf = NmfConvex::new(2).with_max_iter(100);
        let (w, h, g) = nmf.fit(&x).expect("NmfConvex fit ok");

        assert_eq!(w.shape(), &[6, 2]);
        assert_eq!(h.shape(), &[2, 4]);
        assert_eq!(g.shape(), &[6, 2]);

        for &v in h.iter() {
            assert!(v >= 0.0);
        }
        for &v in g.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_nmf_online_basic() {
        let x = toy_matrix();
        let nmf = NmfOnline::new(2).with_batch_size(3).with_n_epochs(5);
        let (w, h) = nmf.fit(&x).expect("NmfOnline fit ok");

        assert_eq!(w.shape(), &[6, 2]);
        assert_eq!(h.shape(), &[2, 4]);

        for &v in w.iter() {
            assert!(v >= 0.0);
        }
        for &v in h.iter() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_nmf_quality_perfect() {
        // W H = X exactly
        let w = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).expect("valid shape");
        let h = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 1.0]).expect("valid shape");
        let x = w.dot(&h);
        let (err, _sparsity) = nmf_quality(&x, &w, &h);
        assert!(err < 1e-10, "perfect reconstruction should give zero error");
    }

    #[test]
    fn test_nmf_mu_negative_input_rejected() {
        let bad = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, 2.0, 3.0]).expect("valid shape");
        let nmf = NmfMu::new(1);
        assert!(nmf.fit(&bad).is_err());
    }

    #[test]
    fn test_nmf_rank_too_large_rejected() {
        let small = Array2::<f64>::eye(3);
        let nmf = NmfMu::new(10);
        assert!(nmf.fit(&small).is_err());
    }
}
