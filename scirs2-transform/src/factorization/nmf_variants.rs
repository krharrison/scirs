//! NMF Variants: Standard NMF, Semi-NMF, Convex NMF, Robust NMF, Deep NMF
//!
//! All algorithms factor a data matrix `X` (shape `n × p`) into two factor matrices
//! under various non-negativity or convexity constraints.
//!
//! ## Algorithms
//!
//! | Struct | Constraint | Loss | Reference |
//! |--------|-----------|------|-----------|
//! | [`NMF`] | `W ≥ 0`, `H ≥ 0` | Frobenius / KL | Lee & Seung 2001 |
//! | [`SemiNMF`] | `H ≥ 0` (W unrestricted) | Frobenius | Ding et al. 2010 |
//! | [`ConvexNMF`] | `W = X S`, `S ≥ 0` | Frobenius | Ding et al. 2010 |
//! | [`RobustNMF`] | `W ≥ 0`, `H ≥ 0` | L2,1 (outlier-robust) | Kong et al. 2011 |
//! | [`DeepNMF`] | `W1 ≥ 0`, `W2 ≥ 0`, `H ≥ 0` | Frobenius | Song et al. 2013 |
//!
//! ## References
//!
//! - Lee, D. D., & Seung, H. S. (2001). Algorithms for NMF. NIPS.
//! - Ding, C., Li, T., & Jordan, M. I. (2010). Convex and semi-NMF. TPAMI, 32(1):45–55.
//! - Kong, D., Ding, C., & Huang, H. (2011). Robust NMF. CVPR.
//! - Song, H. A., & Lee, S. Y. (2013). Hierarchical representation using NMF. ICONIP.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::{Rng, RngExt};

// ─── constants ────────────────────────────────────────────────────────────────
const EPS: f64 = 1e-10;

// ─── generic helpers ─────────────────────────────────────────────────────────

/// Ensure all entries are ≥ `EPS`.
fn clip_nonneg(a: &mut Array2<f64>) {
    a.mapv_inplace(|v| v.max(EPS));
}

/// Frobenius norm squared: `||A - B||_F^2`.
fn frob2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Matrix multiply `A (m×k)` × `B (k×n)` → `C (m×n)`.
fn mm(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    assert_eq!(b.nrows(), k, "mm: inner dimension mismatch");
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for l in 0..k {
            for j in 0..n {
                c[[i, j]] += a[[i, l]] * b[[l, j]];
            }
        }
    }
    c
}

/// Matrix multiply `A^T (k×m)` × `B (m×n)` → `(k×n)`.
fn mm_at_b(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let k = a.ncols();
    let m = a.nrows();
    let n = b.ncols();
    assert_eq!(b.nrows(), m);
    let mut c = Array2::<f64>::zeros((k, n));
    for i in 0..k {
        for l in 0..m {
            for j in 0..n {
                c[[i, j]] += a[[l, i]] * b[[l, j]];
            }
        }
    }
    c
}

/// Matrix multiply `A (m×k)` × `B^T (n×k)` → `(m×n)`.
fn mm_a_bt(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.nrows();
    assert_eq!(b.ncols(), k);
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for l in 0..k {
            for j in 0..n {
                c[[i, j]] += a[[i, l]] * b[[j, l]];
            }
        }
    }
    c
}

/// Initialise a matrix with random uniform values in `[0, scale]`.
fn rand_nonneg(nrows: usize, ncols: usize, scale: f64) -> Array2<f64> {
    let mut rng = scirs2_core::random::rng();
    let mut a = Array2::<f64>::zeros((nrows, ncols));
    for i in 0..nrows {
        for j in 0..ncols {
            a[[i, j]] = scale * rng.gen_range(0.0..1.0_f64);
        }
    }
    a
}

/// Initialise a matrix with random values (possibly negative) in `[-scale, scale]`.
fn rand_signed(nrows: usize, ncols: usize, scale: f64) -> Array2<f64> {
    let mut rng = scirs2_core::random::rng();
    let mut a = Array2::<f64>::zeros((nrows, ncols));
    for i in 0..nrows {
        for j in 0..ncols {
            a[[i, j]] = scale * rng.gen_range(-1.0..1.0_f64);
        }
    }
    a
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Standard NMF
// ─────────────────────────────────────────────────────────────────────────────

/// Divergence measure for NMF objective.
#[derive(Debug, Clone, PartialEq)]
pub enum NmfDivergence {
    /// Frobenius (squared Euclidean) norm: `||X - WH||_F^2`.
    Frobenius,
    /// Generalised Kullback-Leibler divergence: `sum X log(X/(WH)) - X + WH`.
    KullbackLeibler,
}

/// Standard Non-negative Matrix Factorization.
///
/// Factorises `X (n × p)` as `W (n × k) · H (k × p)` with `W ≥ 0`, `H ≥ 0`.
///
/// Uses multiplicative update rules of Lee & Seung (2001) for both
/// Frobenius and KL divergences.
#[derive(Debug, Clone)]
pub struct NMF {
    /// Number of components (latent rank `k`).
    pub n_components: usize,
    /// Divergence measure.
    pub divergence: NmfDivergence,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on relative reconstruction error change.
    pub tol: f64,
    /// Fitted basis matrix `W`, shape `(n_samples, n_components)`.
    pub w: Option<Array2<f64>>,
    /// Fitted coefficient matrix `H`, shape `(n_components, n_features)`.
    pub h: Option<Array2<f64>>,
    /// Reconstruction errors per iteration.
    pub reconstruction_errors: Vec<f64>,
}

impl NMF {
    /// Create a new NMF instance.
    pub fn new(n_components: usize, divergence: NmfDivergence, max_iter: usize, tol: f64) -> Self {
        Self {
            n_components,
            divergence,
            max_iter,
            tol,
            w: None,
            h: None,
            reconstruction_errors: Vec::new(),
        }
    }

    /// Fit the model and return `(W, H)`.
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where
        S: Data<Elem = f64>,
    {
        let x = x.to_owned();
        let (n, p) = (x.nrows(), x.ncols());
        let k = self.n_components;
        if k == 0 || k > n.min(p) {
            return Err(TransformError::InvalidInput(format!(
                "n_components must be in 1..=min({n},{p}), got {k}"
            )));
        }
        if x.iter().any(|&v| v < 0.0) {
            return Err(TransformError::InvalidInput(
                "NMF requires non-negative input matrix".into(),
            ));
        }

        let scale = (x.iter().cloned().fold(0.0_f64, f64::max) / k as f64).sqrt();
        let mut w = rand_nonneg(n, k, scale.max(EPS));
        let mut h = rand_nonneg(k, p, scale.max(EPS));
        clip_nonneg(&mut w);
        clip_nonneg(&mut h);

        self.reconstruction_errors.clear();
        let mut prev_err = f64::INFINITY;

        for _ in 0..self.max_iter {
            match self.divergence {
                NmfDivergence::Frobenius => {
                    // H ← H * (W^T X) / (W^T W H + eps)
                    let wt_x = mm_at_b(&w, &x);
                    let wt_wh = mm_at_b(&w, &mm(&w, &h));
                    for i in 0..k {
                        for j in 0..p {
                            h[[i, j]] *= wt_x[[i, j]] / (wt_wh[[i, j]] + EPS);
                        }
                    }
                    clip_nonneg(&mut h);

                    // W ← W * (X H^T) / (W H H^T + eps)
                    let x_ht = mm_a_bt(&x, &h);
                    let whht = mm_a_bt(&mm(&w, &h), &h);
                    for i in 0..n {
                        for j in 0..k {
                            w[[i, j]] *= x_ht[[i, j]] / (whht[[i, j]] + EPS);
                        }
                    }
                    clip_nonneg(&mut w);
                }
                NmfDivergence::KullbackLeibler => {
                    let wh = mm(&w, &h);
                    // H ← H * (W^T (X / WH)) / (sum_n W)
                    let x_over_wh = Array2::from_shape_fn((n, p), |(i, j)| {
                        x[[i, j]] / (wh[[i, j]] + EPS)
                    });
                    let numerator_h = mm_at_b(&w, &x_over_wh);
                    let sum_w = w.sum_axis(scirs2_core::ndarray::Axis(0)); // shape (k,)
                    for i in 0..k {
                        for j in 0..p {
                            h[[i, j]] *= numerator_h[[i, j]] / (sum_w[i] + EPS);
                        }
                    }
                    clip_nonneg(&mut h);

                    let wh = mm(&w, &h);
                    let x_over_wh2 = Array2::from_shape_fn((n, p), |(i, j)| {
                        x[[i, j]] / (wh[[i, j]] + EPS)
                    });
                    let numerator_w = mm_a_bt(&x_over_wh2, &h);
                    let sum_h = h.sum_axis(scirs2_core::ndarray::Axis(1)); // shape (k,)
                    for i in 0..n {
                        for j in 0..k {
                            w[[i, j]] *= numerator_w[[i, j]] / (sum_h[j] + EPS);
                        }
                    }
                    clip_nonneg(&mut w);
                }
            }

            let wh = mm(&w, &h);
            let err = frob2(&x, &wh).sqrt();
            self.reconstruction_errors.push(err);
            let delta = (prev_err - err).abs() / (prev_err + EPS);
            if delta < self.tol {
                break;
            }
            prev_err = err;
        }

        self.w = Some(w.clone());
        self.h = Some(h.clone());
        Ok((w, h))
    }

    /// Transform new data `X_new` to latent representation `H_new` keeping `W` fixed.
    ///
    /// Minimises `||X_new - W H_new||_F^2` w.r.t. `H_new ≥ 0`.
    pub fn transform<S>(&self, x_new: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data<Elem = f64>,
    {
        let w = self
            .w
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("NMF not fitted".into()))?;
        let x = x_new.to_owned();
        let (n, p) = (x.nrows(), x.ncols());
        let k = self.n_components;
        if p != w.nrows() {
            return Err(TransformError::DimensionMismatch(
                "Feature dimension mismatch in transform".into(),
            ));
        }
        let scale = (x.iter().cloned().fold(0.0_f64, f64::max) / k as f64).sqrt().max(EPS);
        let mut h = rand_nonneg(k, p, scale);
        for _ in 0..200 {
            let wt_x = mm_at_b(w, &x);
            let wt_wh = mm_at_b(w, &mm(w, &h));
            for i in 0..k {
                for j in 0..p {
                    h[[i, j]] *= wt_x[[i, j]] / (wt_wh[[i, j]] + EPS);
                }
            }
            clip_nonneg(&mut h);
        }
        Ok(h)
    }

    /// Reconstruct data from latent codes.
    pub fn inverse_transform(&self, h: &Array2<f64>) -> Result<Array2<f64>> {
        let w = self
            .w
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("NMF not fitted".into()))?;
        Ok(mm(w, h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Semi-NMF
// ─────────────────────────────────────────────────────────────────────────────

/// Semi-NMF: `X ≈ W H` where `H ≥ 0` but `W` can be negative.
///
/// Update rules from Ding et al. (2010):
/// - `H ← H * (W^T X)+ / ((W^T X)- + W^T W H)` (multiplicative, element-wise)
///   where `(A)+ = max(A, 0)` and `(A)- = max(-A, 0)`.
/// - `W ← X H^T (H H^T)^{-1}` (unconstrained least-squares).
#[derive(Debug, Clone)]
pub struct SemiNMF {
    /// Number of components.
    pub n_components: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Fitted W (unrestricted), shape `(n_samples, n_components)`.
    pub w: Option<Array2<f64>>,
    /// Fitted H (non-negative), shape `(n_components, n_features)`.
    pub h: Option<Array2<f64>>,
}

impl SemiNMF {
    /// Create a new SemiNMF instance.
    pub fn new(n_components: usize, max_iter: usize, tol: f64) -> Self {
        Self {
            n_components,
            max_iter,
            tol,
            w: None,
            h: None,
        }
    }

    /// Fit and return `(W, H)`.
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where
        S: Data<Elem = f64>,
    {
        let x = x.to_owned();
        let (n, p) = (x.nrows(), x.ncols());
        let k = self.n_components;
        if k == 0 || k > n.min(p) {
            return Err(TransformError::InvalidInput(format!(
                "n_components must be in 1..=min({n},{p}), got {k}"
            )));
        }

        let scale = 1.0 / (k as f64).sqrt();
        let mut w = rand_signed(n, k, scale);
        let mut h = rand_nonneg(k, p, scale);

        let mut prev_err = f64::INFINITY;

        for _ in 0..self.max_iter {
            // Update H: H ← H * max(W^T X, 0) / (max(-(W^T X), 0) + W^T W H + eps)
            let wt_x = mm_at_b(&w, &x); // (k, p)
            let wt_w = mm_at_b(&w, &w); // (k, k)
            let wt_wh = mm(&wt_w, &h); // (k, p)
            for i in 0..k {
                for j in 0..p {
                    let pos = wt_x[[i, j]].max(0.0);
                    let neg = (-wt_x[[i, j]]).max(0.0);
                    h[[i, j]] *= (pos + EPS) / (neg + wt_wh[[i, j]] + EPS);
                }
            }
            clip_nonneg(&mut h);

            // Update W: W ← X H^T (H H^T)^{-1}  (least squares, unconstrained)
            let x_ht = mm_a_bt(&x, &h); // (n, k)
            let hht = mm_a_bt(&h, &h); // (k, k)
            // Solve W hht = x_ht => W = x_ht hht^{-1}
            // For small k, direct solve via pseudo-inverse is fine
            let hht_inv = pseudo_inv_small(&hht)?;
            w = mm(&x_ht, &hht_inv);

            let err = frob2(&x, &mm(&w, &h)).sqrt();
            let delta = (prev_err - err).abs() / (prev_err + EPS);
            if delta < self.tol {
                break;
            }
            prev_err = err;
        }

        self.w = Some(w.clone());
        self.h = Some(h.clone());
        Ok((w, h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Convex NMF
// ─────────────────────────────────────────────────────────────────────────────

/// Convex NMF: `X ≈ X S H` where `S ≥ 0`, `H ≥ 0`.
///
/// The columns of `W = X S` are convex combinations of the data points,
/// so the basis vectors lie in the convex hull of the data.
///
/// Update rules (Ding et al. 2010, Theorem 7):
/// ```text
/// S ← S * (X^T X H^T)+ / ((X^T X H^T)- + X^T X S H H^T + eps)
/// H ← H * (S^T X^T X)+ / ((S^T X^T X)- + H S^T X^T X S + eps)
/// ```
#[derive(Debug, Clone)]
pub struct ConvexNMF {
    /// Number of components.
    pub n_components: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Fitted S (non-negative), shape `(n_samples, n_components)`.
    pub s: Option<Array2<f64>>,
    /// Fitted H (non-negative), shape `(n_components, n_samples)`.
    pub h: Option<Array2<f64>>,
}

impl ConvexNMF {
    /// Create a new ConvexNMF instance.
    pub fn new(n_components: usize, max_iter: usize, tol: f64) -> Self {
        Self {
            n_components,
            max_iter,
            tol,
            s: None,
            h: None,
        }
    }

    /// Fit and return `(W, H)` where `W = X S`.
    ///
    /// Both `S ≥ 0` and `H ≥ 0`.  Columns of `W` are convex combinations of data.
    pub fn fit_transform<S2>(&mut self, x: &ArrayBase<S2, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where
        S2: Data<Elem = f64>,
    {
        let x = x.to_owned();
        let (n, p) = (x.nrows(), x.ncols());
        let k = self.n_components;
        if k == 0 || k > n {
            return Err(TransformError::InvalidInput(format!(
                "n_components must be in 1..={n}, got {k}"
            )));
        }

        let scale = 1.0 / (k as f64).sqrt();
        let mut s = rand_nonneg(n, k, scale); // n × k
        let mut h = rand_nonneg(k, n, scale); // k × n  (codes over samples)

        // Pre-compute X^T X  (p × p, but note we treat X as n×p)
        // Actually convex NMF works in sample space: X ≈ X S H  => size (n×p)≈(n×p)
        // W = X S: (n×p)(p??  --  X: n×p, S: n×k, so W = X^T S is p×k... 
        // Ding et al. formulation: X≈XSH where X:n×p, S:n×k, H:k×n => XSH:n×n×... wrong
        // Correct: X≈WSH where W=XG, G:n×k with G≥0, H:k×p with H≥0
        // Let us use G (n×k) and H (k×p) so X(n×p) ≈ X(n×p) G(n×k) ??? 
        // The formulation is: X ≈ (X G) H, where X G is n×k and H is k×p.
        // G≥0 ensures each basis is a combo of data rows. H≥0.
        // Let G (n×k), H (k×p). X^T X is p×p... big for high-dim.
        // Use "kernel" form: K = X X^T (n×n), then (XG) = columns of X weighted by G.
        // Update G: G ← G * (K H^T)+ / ((K H^T)- + K G H H^T + eps)
        // Update H: H ← H * (G^T K)+ / ((G^T K)- + H G^T K G + eps)

        // Re-initialise with proper naming
        let mut g = s; // n × k
        let mut h2 = rand_nonneg(k, p, scale); // k × p

        // K = X X^T (n × n)
        let k_mat = mm_a_bt(&x, &x); // (n, n)

        let mut prev_err = f64::INFINITY;

        for _ in 0..self.max_iter {
            // Update G
            let kht = mm_a_bt(&k_mat, &h2); // (n, k)
            let ghht = mm_a_bt(&mm(&k_mat, &g), &h2); // wait: (n×n)(n×k) = (n×k), then (n×k)(k×p)(p×k)??
            // (K G) H H^T: K(n×n) G(n×k) → (n×k); H(k×p) H^T(p×k) → (k×k); (n×k)(k×k)=(n×k)
            let kg = mm(&k_mat, &g); // (n, k)
            let hht = mm_a_bt(&h2, &h2); // (k, k)
            let kg_hht = mm(&kg, &hht); // (n, k)

            for i in 0..n {
                for j in 0..k {
                    let pos = kht[[i, j]].max(0.0);
                    let neg = (-kht[[i, j]]).max(0.0);
                    g[[i, j]] *= (pos + EPS) / (neg + kg_hht[[i, j]] + EPS);
                }
            }
            clip_nonneg(&mut g);

            // Update H
            let gtk = mm_at_b(&g, &k_mat); // (k, n)
            let gtkg = mm(&gtk, &g); // (k, k)
            let gtkx = mm(&gtk, &x); // (k, p) = G^T K X ... wait K=XX^T so K X = X X^T X
            // Actually we need G^T X (not G^T K) for approximating X directly.
            // Let me redo: X ≈ (XG) H2, XG:(n×k), H2:(k×p)
            // ||X - XG H2||_F^2
            // Update H2: H2 ← H2 * (G^T X^T X)+ / (...)  -- but X^T X is p×p, large
            // Use: ((XG)^T X) = G^T (X^T X) but we avoid X^T X.
            // (XG)^T X = G^T X^T X  but K=X X^T so: G^T K X ... not helpful.
            // Better: numerator = (XG)^T X = G^T X^T X (k×p) -- compute as mm_at_b(g, mm_at_b(x, x)) ??
            // Actually (XG)^T X: (XG) is n×k, X is n×p, so (XG)^T X is k×p.
            // = G^T (X^T X) ... but let's just compute it directly:
            let xg = mm(&x, &g); // (n, k)
            let xgt_x = mm_at_b(&xg, &x); // (k, p) -- this is (XG)^T X
            // denominator: (XG)^T (XG) H2 = xg^T xg H2
            let xgt_xg = mm_at_b(&xg, &xg); // (k, k)
            let xgt_xg_h = mm(&xgt_xg, &h2); // (k, p)

            for i in 0..k {
                for j in 0..p {
                    let pos = xgt_x[[i, j]].max(0.0);
                    let neg = (-xgt_x[[i, j]]).max(0.0);
                    h2[[i, j]] *= (pos + EPS) / (neg + xgt_xg_h[[i, j]] + EPS);
                }
            }
            clip_nonneg(&mut h2);

            // Reconstruction error: ||X - XG H2||_F
            let x_hat = mm(&mm(&x, &g), &h2);
            let err = frob2(&x, &x_hat).sqrt();
            let delta = (prev_err - err).abs() / (prev_err + EPS);
            if delta < self.tol {
                break;
            }
            prev_err = err;
        }

        let w = mm(&x, &g); // n × k
        self.s = Some(g);
        self.h = Some(h2.clone());
        Ok((w, h2))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Robust NMF
// ─────────────────────────────────────────────────────────────────────────────

/// Robust NMF with L2,1 loss.
///
/// Minimises `sum_i ||x_i - W h_i||_2` (sum of L2 norms per sample),
/// which is robust to sample-wise outliers.
///
/// The reweighted multiplicative updates use diagonal weight matrix `D`
/// where `D_{ii} = 1 / (2 ||x_i - W h_i||_2 + eps)`.
///
/// Reference: Kong, D., Ding, C., & Huang, H. (2011). Robust NMF using L2,1 norm. CVPR.
#[derive(Debug, Clone)]
pub struct RobustNMF {
    /// Number of components.
    pub n_components: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Fitted W, shape `(n_features, n_components)`.
    pub w: Option<Array2<f64>>,
    /// Fitted H, shape `(n_components, n_samples)`.
    pub h: Option<Array2<f64>>,
}

impl RobustNMF {
    /// Create a new RobustNMF instance.
    pub fn new(n_components: usize, max_iter: usize, tol: f64) -> Self {
        Self {
            n_components,
            max_iter,
            tol,
            w: None,
            h: None,
        }
    }

    /// Fit and return `(W, H)`.
    ///
    /// `X` has shape `(n_samples, n_features)`.
    /// `W` has shape `(n_features, n_components)`, `H` has shape `(n_components, n_samples)`.
    pub fn fit_transform<S2>(&mut self, x: &ArrayBase<S2, Ix2>) -> Result<(Array2<f64>, Array2<f64>)>
    where
        S2: Data<Elem = f64>,
    {
        let x_in = x.to_owned();
        // Internally we work with X^T: p × n  (features × samples)
        let (n, p) = (x_in.nrows(), x_in.ncols());
        let k = self.n_components;
        if k == 0 || k > p.min(n) {
            return Err(TransformError::InvalidInput(format!(
                "n_components must be in 1..=min({n},{p}), got {k}"
            )));
        }

        let scale = 1.0 / (k as f64).sqrt();
        // W: p × k,  H: k × n
        let mut w = rand_nonneg(p, k, scale);
        let mut h = rand_nonneg(k, n, scale);

        // xt: p × n
        let mut xt = Array2::<f64>::zeros((p, n));
        for i in 0..n {
            for j in 0..p {
                xt[[j, i]] = x_in[[i, j]];
            }
        }

        let mut prev_err = f64::INFINITY;

        for _ in 0..self.max_iter {
            // Compute residuals R = X^T - W H  (p × n)
            let wh = mm(&w, &h); // p × n
            let mut d_diag = vec![0.0f64; n];
            for col in 0..n {
                let mut sum_sq = 0.0;
                for row in 0..p {
                    let r = xt[[row, col]] - wh[[row, col]];
                    sum_sq += r * r;
                }
                d_diag[col] = 1.0 / (2.0 * sum_sq.sqrt() + EPS);
            }

            // Update H: H ← H * (W^T X^T D)+ / (W^T W H D + eps)
            // where D is diagonal with d_diag
            // W^T X^T D: k×p @ p×n scaled by D => k×n with col j scaled by d_diag[j]
            let wt_xt = mm_at_b(&w, &xt); // k × n
            let wt_w = mm_at_b(&w, &w); // k × k
            let wt_wh = mm(&wt_w, &h); // k × n

            for i in 0..k {
                for j in 0..n {
                    let num = wt_xt[[i, j]] * d_diag[j];
                    let den = wt_wh[[i, j]] * d_diag[j] + EPS;
                    h[[i, j]] *= num.max(EPS) / den;
                }
            }
            clip_nonneg(&mut h);

            // Update W: W ← W * (X^T D H^T) / (W H D H^T + eps)
            let hd: Array2<f64> = Array2::from_shape_fn((k, n), |(i, j)| h[[i, j]] * d_diag[j]);
            let xt_d_ht = mm_a_bt(&xt, &hd); // p × k   (X^T D H^T — note D absorbed in hd)
            // Actually X^T D H^T = X^T (D H)^T = (H D)^T X = (hd)^T X wait...
            // X^T: p×n, D: n×n diag, H^T: n×k => X^T D H^T is p×k
            // = xt (n×p)^T ... let me redo: xt is p×n, D is n diag, H is k×n
            // X^T D H^T = xt * diag(d) * H^T = (xt diag(d)) H^T
            let xt_d = Array2::from_shape_fn((p, n), |(i, j)| xt[[i, j]] * d_diag[j]); // p×n
            let xt_d_ht = mm_a_bt(&xt_d, &h); // p×k  (xt_d H^T)
            let whdt = mm_a_bt(&mm(&w, &hd), &h); // W (H D) H^T: p×n×k×n = wrong
            // W H D H^T: w(p×k) h(k×n) diag(d)(n×n) H^T(n×k) = w(p×k) hd(k×n) H^T(n×k)
            let whd_ht = mm_a_bt(&mm(&w, &hd), &h); // (p×k)(k×n) H^T => p×n × n×k = p×k

            for i in 0..p {
                for j in 0..k {
                    w[[i, j]] *= (xt_d_ht[[i, j]] + EPS) / (whd_ht[[i, j]] + EPS);
                }
            }
            clip_nonneg(&mut w);

            // L2,1 loss
            let wh2 = mm(&w, &h);
            let mut err = 0.0f64;
            for col in 0..n {
                let mut sum_sq = 0.0;
                for row in 0..p {
                    let r = xt[[row, col]] - wh2[[row, col]];
                    sum_sq += r * r;
                }
                err += sum_sq.sqrt();
            }
            let delta = (prev_err - err).abs() / (prev_err + EPS);
            if delta < self.tol {
                break;
            }
            prev_err = err;
        }

        self.w = Some(w.clone());
        self.h = Some(h.clone());
        Ok((w, h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Deep NMF
// ─────────────────────────────────────────────────────────────────────────────

/// Two-layer Deep NMF.
///
/// Factorises `X ≈ W1 · W2 · H` with `W1, W2, H ≥ 0`.
///
/// Each layer is updated by solving an NMF sub-problem while keeping the other
/// layers fixed, following Song et al. (2013) "Hierarchical NMF".
///
/// Layer 1: `X ≈ W1 · Z1`   where `Z1 = W2 H`
/// Layer 2: `Z1 ≈ W2 · H`
///
/// Alternating updates cycle through fixing `(W2, H)` and updating `W1`,
/// then fixing `(W1, H)` and updating `W2`, then fixing `(W1, W2)` and
/// updating `H`.
#[derive(Debug, Clone)]
pub struct DeepNMF {
    /// Number of components in layer 1 (hidden dim 1): `k1`.
    pub n_components_1: usize,
    /// Number of components in layer 2 (hidden dim 2): `k2`.
    pub n_components_2: usize,
    /// Maximum alternating iterations.
    pub max_iter: usize,
    /// Inner iterations per layer update.
    pub inner_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Layer-1 basis `W1`, shape `(n_features, k1)`.
    pub w1: Option<Array2<f64>>,
    /// Layer-2 basis `W2`, shape `(k1, k2)`.
    pub w2: Option<Array2<f64>>,
    /// Code matrix `H`, shape `(k2, n_samples)`.
    pub h: Option<Array2<f64>>,
}

impl DeepNMF {
    /// Create a new DeepNMF instance.
    pub fn new(
        n_components_1: usize,
        n_components_2: usize,
        max_iter: usize,
        inner_iter: usize,
        tol: f64,
    ) -> Self {
        Self {
            n_components_1,
            n_components_2,
            max_iter,
            inner_iter,
            tol,
            w1: None,
            w2: None,
            h: None,
        }
    }

    /// Fit and return `(W1, W2, H)`.
    ///
    /// `X` has shape `(n_samples, n_features)`.
    pub fn fit_transform<S2>(
        &mut self,
        x: &ArrayBase<S2, Ix2>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)>
    where
        S2: Data<Elem = f64>,
    {
        let x_in = x.to_owned();
        let (n, p) = (x_in.nrows(), x_in.ncols());
        let k1 = self.n_components_1;
        let k2 = self.n_components_2;

        if k1 == 0 || k1 > p.min(n) {
            return Err(TransformError::InvalidInput(format!(
                "n_components_1 must be in 1..=min({n},{p}), got {k1}"
            )));
        }
        if k2 == 0 || k2 > k1 {
            return Err(TransformError::InvalidInput(format!(
                "n_components_2 must be in 1..={k1}, got {k2}"
            )));
        }

        // Work with X^T: p × n for convention W1(p×k1) W2(k1×k2) H(k2×n)
        let mut xt = Array2::<f64>::zeros((p, n));
        for i in 0..n {
            for j in 0..p {
                xt[[j, i]] = x_in[[i, j]];
            }
        }

        let scale = ((xt.iter().cloned().fold(0.0_f64, f64::max)) / k1 as f64).sqrt();
        let scale = scale.max(EPS);

        let mut w1 = rand_nonneg(p, k1, scale); // p × k1
        let mut w2 = rand_nonneg(k1, k2, scale / (k1 as f64).sqrt()); // k1 × k2
        let mut h = rand_nonneg(k2, n, scale / (k1 as f64 * k2 as f64).sqrt()); // k2 × n

        let mut prev_err = f64::INFINITY;

        for _ in 0..self.max_iter {
            // ── Update W1 (fix W2, H): X^T ≈ W1 (W2 H) ──────────────────────
            let z1 = mm(&w2, &h); // k1 × n  (this is W2 H)
            for _ in 0..self.inner_iter {
                let num = mm_a_bt(&xt, &z1); // p × k1
                let den = mm_a_bt(&mm(&w1, &z1), &z1); // p × k1
                for i in 0..p {
                    for j in 0..k1 {
                        w1[[i, j]] *= (num[[i, j]] + EPS) / (den[[i, j]] + EPS);
                    }
                }
                clip_nonneg(&mut w1);
            }

            // ── Update W2 (fix W1, H): W1^T X^T ≈ W2 H ──────────────────────
            let z2 = mm_at_b(&w1, &xt); // k1 × n  (this is W1^T X^T)
            for _ in 0..self.inner_iter {
                let num = mm_a_bt(&z2, &h); // k1 × k2
                let den = mm_a_bt(&mm(&w2, &h), &h); // k1 × k2
                for i in 0..k1 {
                    for j in 0..k2 {
                        w2[[i, j]] *= (num[[i, j]] + EPS) / (den[[i, j]] + EPS);
                    }
                }
                clip_nonneg(&mut w2);
            }

            // ── Update H (fix W1, W2): (W1 W2)^T X^T ≈ H ────────────────────
            let w1w2 = mm(&w1, &w2); // p × k2
            for _ in 0..self.inner_iter {
                let num = mm_at_b(&w1w2, &xt); // k2 × n
                let den = mm_at_b(&w1w2, &mm(&w1w2, &h)); // k2 × n
                for i in 0..k2 {
                    for j in 0..n {
                        h[[i, j]] *= (num[[i, j]] + EPS) / (den[[i, j]] + EPS);
                    }
                }
                clip_nonneg(&mut h);
            }

            // Reconstruction error
            let x_hat = mm(&mm(&w1, &w2), &h); // p × n
            let err = frob2(&xt, &x_hat).sqrt();
            let delta = (prev_err - err).abs() / (prev_err + EPS);
            if delta < self.tol {
                break;
            }
            prev_err = err;
        }

        self.w1 = Some(w1.clone());
        self.w2 = Some(w2.clone());
        self.h = Some(h.clone());
        Ok((w1, w2, h))
    }

    /// Reconstruct data from the three-factor model.
    pub fn reconstruct(&self) -> Result<Array2<f64>> {
        let w1 = self.w1.as_ref().ok_or_else(|| TransformError::NotFitted("DeepNMF not fitted".into()))?;
        let w2 = self.w2.as_ref().ok_or_else(|| TransformError::NotFitted("DeepNMF not fitted".into()))?;
        let h = self.h.as_ref().ok_or_else(|| TransformError::NotFitted("DeepNMF not fitted".into()))?;
        // Returns p × n; caller may need to transpose
        Ok(mm(&mm(w1, w2), h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Pseudo-inverse of a small square matrix using Tikhonov regularisation.
fn pseudo_inv_small(a: &Array2<f64>) -> Result<Array2<f64>> {
    let k = a.nrows();
    // Regularise: A + lambda I
    let mut ar = a.to_owned();
    let trace: f64 = (0..k).map(|i| a[[i, i]]).sum();
    let lambda = (trace / k as f64 * 1e-6).max(EPS);
    for i in 0..k {
        ar[[i, i]] += lambda;
    }
    invert_small_gauss(&ar)
}

/// Gaussian elimination inverse (used inside pseudo_inv_small).
fn invert_small_gauss(mat: &Array2<f64>) -> Result<Array2<f64>> {
    let k = mat.nrows();
    let mut aug = Array2::<f64>::zeros((k, 2 * k));
    for i in 0..k {
        for j in 0..k {
            aug[[i, j]] = mat[[i, j]];
        }
        aug[[i, k + i]] = 1.0;
    }
    for col in 0..k {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..k {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < EPS {
            return Err(TransformError::ComputationError("Singular matrix in pseudo_inv".into()));
        }
        if max_row != col {
            for j in 0..(2 * k) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }
        let diag = aug[[col, col]];
        for j in 0..(2 * k) {
            aug[[col, j]] /= diag;
        }
        for row in 0..k {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            for j in 0..(2 * k) {
                let v = aug[[col, j]] * factor;
                aug[[row, j]] -= v;
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            inv[[i, j]] = aug[[i, k + j]];
        }
    }
    Ok(inv)
}

/// Compute reconstruction quality metrics `(frobenius_error, sparsity_of_h)`.
///
/// - `frobenius_error` = `||X - WH||_F / ||X||_F`
/// - `sparsity_of_h` = fraction of H entries < 1e-6
pub fn nmf_quality(x: &Array2<f64>, w: &Array2<f64>, h: &Array2<f64>) -> (f64, f64) {
    let wh = mm(w, h);
    let num = frob2(x, &wh).sqrt();
    let den = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let frobenius_error = num / (den + EPS);
    let total = h.len();
    let sparse_count = h.iter().filter(|&&v| v < 1e-6).count();
    let sparsity = sparse_count as f64 / total as f64;
    (frobenius_error, sparsity)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_nonneg_data(n: usize, p: usize, k: usize) -> Array2<f64> {
        let mut rng = scirs2_core::random::rng();
        let mut w = Array2::<f64>::zeros((n, k));
        let mut h = Array2::<f64>::zeros((k, p));
        for i in 0..n {
            for j in 0..k {
                w[[i, j]] = rng.gen_range(0.0..2.0);
            }
        }
        for i in 0..k {
            for j in 0..p {
                h[[i, j]] = rng.gen_range(0.0..2.0);
            }
        }
        mm(&w, &h)
    }

    #[test]
    fn test_nmf_frobenius() {
        let x = make_nonneg_data(20, 15, 3);
        let mut nmf = NMF::new(3, NmfDivergence::Frobenius, 200, 1e-4);
        let (w, h) = nmf.fit_transform(&x).expect("NMF fit failed");
        assert_eq!(w.shape(), &[20, 3]);
        assert_eq!(h.shape(), &[3, 15]);
        assert!(w.iter().all(|&v| v >= 0.0));
        assert!(h.iter().all(|&v| v >= 0.0));
        let (err, _) = nmf_quality(&x, &w, &h);
        assert!(err < 0.5, "Reconstruction error {err} too large");
    }

    #[test]
    fn test_nmf_kl() {
        let x = make_nonneg_data(15, 10, 2);
        let mut nmf = NMF::new(2, NmfDivergence::KullbackLeibler, 100, 1e-4);
        let (w, h) = nmf.fit_transform(&x).expect("NMF KL fit failed");
        assert_eq!(w.shape(), &[15, 2]);
        assert_eq!(h.shape(), &[2, 10]);
        assert!(w.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_semi_nmf() {
        let mut rng = scirs2_core::random::rng();
        let mut x = Array2::<f64>::zeros((15, 10));
        for i in 0..15 {
            for j in 0..10 {
                x[[i, j]] = rng.gen_range(-1.0..2.0);
            }
        }
        let mut model = SemiNMF::new(3, 100, 1e-4);
        let (w, h) = model.fit_transform(&x).expect("SemiNMF failed");
        assert_eq!(w.shape(), &[15, 3]);
        assert_eq!(h.shape(), &[3, 10]);
        assert!(h.iter().all(|&v| v >= 0.0), "H must be non-negative");
    }

    #[test]
    fn test_convex_nmf() {
        let x = make_nonneg_data(12, 8, 2);
        let mut model = ConvexNMF::new(2, 50, 1e-4);
        let (w, h) = model.fit_transform(&x).expect("ConvexNMF failed");
        assert_eq!(w.shape(), &[12, 2]);
        assert!(h.iter().all(|&v| v >= 0.0), "H must be non-negative");
        assert!(w.iter().all(|&v| v >= 0.0), "W must be non-negative");
    }

    #[test]
    fn test_robust_nmf() {
        let x = make_nonneg_data(20, 8, 2);
        let mut model = RobustNMF::new(2, 50, 1e-4);
        let (w, h) = model.fit_transform(&x).expect("RobustNMF failed");
        assert_eq!(w.shape(), &[8, 2]);
        assert_eq!(h.shape(), &[2, 20]);
        assert!(w.iter().all(|&v| v >= 0.0));
        assert!(h.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_deep_nmf() {
        let x = make_nonneg_data(20, 10, 3);
        let mut model = DeepNMF::new(4, 2, 30, 5, 1e-4);
        let (w1, w2, h) = model.fit_transform(&x).expect("DeepNMF failed");
        assert_eq!(w1.shape(), &[10, 4]);
        assert_eq!(w2.shape(), &[4, 2]);
        assert_eq!(h.shape(), &[2, 20]);
        assert!(w1.iter().all(|&v| v >= 0.0));
        assert!(w2.iter().all(|&v| v >= 0.0));
        assert!(h.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_nmf_quality() {
        let x = make_nonneg_data(10, 8, 2);
        let mut nmf = NMF::new(2, NmfDivergence::Frobenius, 100, 1e-4);
        let (w, h) = nmf.fit_transform(&x).expect("NMF fit failed");
        let (err, sparsity) = nmf_quality(&x, &w, &h);
        assert!(err.is_finite());
        assert!(sparsity >= 0.0 && sparsity <= 1.0);
    }
}
