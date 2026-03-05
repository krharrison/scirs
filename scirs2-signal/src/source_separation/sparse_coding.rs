//! Sparse coding and dictionary learning.
//!
//! Provides OMP, ISTA-Lasso, and K-SVD dictionary learning.

use crate::error::{SignalError, SignalResult};

const EPS: f64 = 1e-12;

// ──────────────────────────────────────────────────────────────────────────────
// Soft / Hard threshold utilities (public)
// ──────────────────────────────────────────────────────────────────────────────

/// Soft-thresholding operator: `S_λ(x) = sign(x) * max(|x| - λ, 0)`.
pub fn soft_threshold(v: &[f64], lambda: f64) -> Vec<f64> {
    v.iter()
        .map(|&x| {
            let abs = x.abs();
            if abs <= lambda {
                0.0
            } else {
                x.signum() * (abs - lambda)
            }
        })
        .collect()
}

/// Hard-thresholding operator: keep the `k` entries with largest absolute value.
pub fn hard_threshold(v: &[f64], k: usize) -> Vec<f64> {
    if k == 0 || v.is_empty() {
        return vec![0.0; v.len()];
    }
    // Find the k-th largest magnitude threshold
    let mut magnitudes: Vec<f64> = v.iter().map(|&x| x.abs()).collect();
    magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = if k < magnitudes.len() {
        magnitudes[k - 1]
    } else {
        0.0
    };
    v.iter()
        .map(|&x| if x.abs() >= threshold - EPS { x } else { 0.0 })
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Dot product of two slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// L2 norm of a slice.
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Subtract `alpha * atom` from `residual` in-place.
fn axpy(residual: &mut [f64], atom: &[f64], alpha: f64) {
    for (r, &d) in residual.iter_mut().zip(atom.iter()) {
        *r -= alpha * d;
    }
}

/// Least-squares projection `min ||D_S c - signal||` on the support S.
/// D_S columns are selected by `support` indices.
/// Uses normal equations (D_S^T D_S) c = D_S^T signal.
fn least_squares_on_support(
    signal: &[f64],
    dictionary: &[Vec<f64>],
    support: &[usize],
) -> SignalResult<Vec<f64>> {
    let k = support.len();
    let n = signal.len();
    if k == 0 {
        return Ok(vec![]);
    }
    // Build D_S  (n × k) as row-major vec for clarity
    // D_S^T D_S  is (k × k)
    let mut gram = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let s = dot(&dictionary[support[i]], &dictionary[support[j]]);
            gram[i * k + j] = s;
        }
    }
    // D_S^T signal  (k)
    let mut rhs: Vec<f64> = support
        .iter()
        .map(|&idx| dot(&dictionary[idx], signal))
        .collect();

    // Gaussian elimination
    for col in 0..k {
        let mut max_val = gram[col * k + col].abs();
        let mut max_row = col;
        for row in (col + 1)..k {
            let v = gram[row * k + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(SignalError::ComputationError(
                "LS on support: singular Gram matrix".into(),
            ));
        }
        if max_row != col {
            for j in 0..k {
                gram.swap(col * k + j, max_row * k + j);
            }
            rhs.swap(col, max_row);
        }
        let pivot = gram[col * k + col];
        for row in (col + 1)..k {
            let factor = gram[row * k + col] / pivot;
            for j in col..k {
                let sub = factor * gram[col * k + j];
                gram[row * k + j] -= sub;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut coeff = vec![0.0_f64; k];
    for row in (0..k).rev() {
        let mut val = rhs[row];
        for j in (row + 1)..k {
            val -= gram[row * k + j] * coeff[j];
        }
        coeff[row] = val / gram[row * k + row];
    }
    Ok(coeff)
}

/// Reconstruct signal from support and coefficients.
fn reconstruct_from_support(
    dictionary: &[Vec<f64>],
    support: &[usize],
    coeff: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    for (idx_pos, &atom_idx) in support.iter().enumerate() {
        let c = coeff[idx_pos];
        for (o, &d) in out.iter_mut().zip(dictionary[atom_idx].iter()) {
            *o += c * d;
        }
    }
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// OMP
// ──────────────────────────────────────────────────────────────────────────────

/// Orthogonal Matching Pursuit encoder.
pub struct OMP {
    /// Maximum number of non-zero coefficients.
    pub max_nnz: usize,
}

impl OMP {
    /// Create a new OMP encoder with `max_nnz` maximum non-zeros.
    pub fn new(max_nnz: usize) -> Self {
        Self { max_nnz }
    }

    /// Encode `signal` using `dictionary` (list of atoms, each same length as signal).
    ///
    /// Returns a sparse code vector of length `dictionary.len()`.
    pub fn encode(&self, signal: &[f64], dictionary: &[Vec<f64>]) -> SignalResult<Vec<f64>> {
        let n_atoms = dictionary.len();
        if n_atoms == 0 {
            return Err(SignalError::InvalidArgument("Empty dictionary".into()));
        }
        let sig_len = signal.len();
        for (i, atom) in dictionary.iter().enumerate() {
            if atom.len() != sig_len {
                return Err(SignalError::DimensionMismatch(format!(
                    "Atom {i} length {} ≠ signal length {sig_len}",
                    atom.len()
                )));
            }
        }

        let k = self.max_nnz.min(n_atoms);
        let mut residual = signal.to_vec();
        let mut support: Vec<usize> = Vec::with_capacity(k);
        let mut code = vec![0.0_f64; n_atoms];

        for _step in 0..k {
            // Select atom most correlated with residual
            let mut best_idx = 0;
            let mut best_corr = 0.0_f64;
            for (j, atom) in dictionary.iter().enumerate() {
                if support.contains(&j) {
                    continue;
                }
                let atom_norm = norm2(atom);
                if atom_norm < EPS {
                    continue;
                }
                let corr = dot(&residual, atom).abs() / atom_norm;
                if corr > best_corr {
                    best_corr = corr;
                    best_idx = j;
                }
            }
            support.push(best_idx);

            // Orthogonal projection onto span of selected atoms
            match least_squares_on_support(signal, dictionary, &support) {
                Ok(coeff) => {
                    // Update code and residual
                    let approx = reconstruct_from_support(dictionary, &support, &coeff, sig_len);
                    for (r, (&s, &a)) in residual.iter_mut().zip(signal.iter().zip(approx.iter())) {
                        *r = s - a;
                    }
                    for (pos, &idx) in support.iter().enumerate() {
                        code[idx] = coeff[pos];
                    }
                }
                Err(_) => {
                    // Fallback: simple subtraction
                    let atom = &dictionary[best_idx];
                    let atom_norm_sq = dot(atom, atom).max(EPS);
                    let alpha = dot(&residual, atom) / atom_norm_sq;
                    code[best_idx] += alpha;
                    axpy(&mut residual, atom, alpha);
                }
            }

            if norm2(&residual) < EPS {
                break;
            }
        }

        Ok(code)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ISTA Lasso
// ──────────────────────────────────────────────────────────────────────────────

/// ISTA (Iterative Shrinkage-Thresholding Algorithm) for L1-regularized sparse coding.
pub struct Lasso {
    /// L1 regularization weight.
    pub lambda: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Lasso {
    /// Create a new Lasso encoder.
    pub fn new(lambda: f64, max_iter: usize, tol: f64) -> Self {
        Self { lambda, max_iter, tol }
    }

    /// Estimate Lipschitz constant L = ||D^T D||_2 ≈ max eigenvalue (power method).
    fn lipschitz(dictionary: &[Vec<f64>]) -> f64 {
        let n_atoms = dictionary.len();
        if n_atoms == 0 {
            return 1.0;
        }
        // Gram matrix G = D^T D  (n_atoms × n_atoms)
        // Power iteration to estimate spectral norm
        let mut v = vec![1.0_f64 / (n_atoms as f64).sqrt(); n_atoms];
        for _ in 0..50 {
            // Gv
            let mut gv = vec![0.0_f64; n_atoms];
            for i in 0..n_atoms {
                let di = &dictionary[i];
                for j in 0..n_atoms {
                    let gij: f64 = di.iter().zip(dictionary[j].iter()).map(|(&a, &b)| a * b).sum();
                    gv[i] += gij * v[j];
                }
            }
            let n = norm2(&gv).max(EPS);
            for val in &mut gv {
                *val /= n;
            }
            v = gv;
        }
        // Rayleigh quotient
        let mut gv = vec![0.0_f64; n_atoms];
        for i in 0..n_atoms {
            let di = &dictionary[i];
            for j in 0..n_atoms {
                let gij: f64 = di.iter().zip(dictionary[j].iter()).map(|(&a, &b)| a * b).sum();
                gv[i] += gij * v[j];
            }
        }
        dot(&v, &gv).max(1.0)
    }

    /// Encode `signal` via ISTA.
    ///
    /// Update: `x ← S_{λ/L}(x + (1/L) D^T (y - D x))`
    pub fn encode(&self, signal: &[f64], dictionary: &[Vec<f64>]) -> SignalResult<Vec<f64>> {
        let n_atoms = dictionary.len();
        if n_atoms == 0 {
            return Err(SignalError::InvalidArgument("Empty dictionary".into()));
        }
        let sig_len = signal.len();
        for (i, atom) in dictionary.iter().enumerate() {
            if atom.len() != sig_len {
                return Err(SignalError::DimensionMismatch(format!(
                    "Atom {i} length {} ≠ signal length {sig_len}",
                    atom.len()
                )));
            }
        }

        let l = Self::lipschitz(dictionary);
        let step = 1.0 / l;
        let thresh = self.lambda * step;

        let mut x = vec![0.0_f64; n_atoms];

        for _iter in 0..self.max_iter {
            // residual = y - D x
            let mut residual = signal.to_vec();
            for (j, atom) in dictionary.iter().enumerate() {
                axpy(&mut residual, atom, x[j]);
            }
            // gradient = -D^T residual  →  x_new = x + step * D^T residual
            let grad: Vec<f64> = dictionary
                .iter()
                .map(|atom| dot(atom, &residual))
                .collect();

            let x_new: Vec<f64> = x
                .iter()
                .zip(grad.iter())
                .map(|(&xi, &gi)| xi + step * gi)
                .collect();
            let x_soft = soft_threshold(&x_new, thresh);

            // Check convergence
            let diff: f64 = x_soft.iter().zip(x.iter()).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>().sqrt();
            x = x_soft;
            if diff < self.tol {
                break;
            }
        }

        Ok(x)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// K-SVD Dictionary Learning
// ──────────────────────────────────────────────────────────────────────────────

/// K-SVD dictionary learning.
pub struct DictionaryLearning {
    /// Number of dictionary atoms.
    pub n_atoms: usize,
    /// Maximum sparsity per signal (used by OMP).
    pub sparsity: usize,
    /// Number of K-SVD iterations.
    pub max_iter: usize,
}

impl DictionaryLearning {
    /// Create a new DictionaryLearning instance.
    pub fn new(n_atoms: usize, sparsity: usize, max_iter: usize) -> Self {
        Self { n_atoms, sparsity, max_iter }
    }

    /// Fit a dictionary to the training signals.
    ///
    /// Returns the learned dictionary (list of atoms, each of length `signal_len`).
    pub fn fit(&self, signals: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
        let (dict, _) = self.fit_transform(signals)?;
        Ok(dict)
    }

    /// Fit and return `(dictionary, codes)`.
    pub fn fit_transform(
        &self,
        signals: &[Vec<f64>],
    ) -> SignalResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let n_signals = signals.len();
        if n_signals == 0 {
            return Err(SignalError::InvalidArgument("Empty signals".into()));
        }
        let sig_len = signals[0].len();
        if sig_len == 0 {
            return Err(SignalError::InvalidArgument("Zero-length signals".into()));
        }
        for (i, s) in signals.iter().enumerate() {
            if s.len() != sig_len {
                return Err(SignalError::DimensionMismatch(format!(
                    "Signal {i} length {} ≠ {sig_len}",
                    s.len()
                )));
            }
        }
        if self.n_atoms == 0 {
            return Err(SignalError::InvalidArgument("n_atoms must be > 0".into()));
        }
        if self.sparsity == 0 {
            return Err(SignalError::InvalidArgument("sparsity must be > 0".into()));
        }

        // Initialise dictionary: select random signals and normalise
        let mut dict = self.init_dict(signals, sig_len);

        let omp = OMP::new(self.sparsity);
        let mut codes: Vec<Vec<f64>> = vec![vec![0.0; self.n_atoms]; n_signals];

        for _iter in 0..self.max_iter {
            // ── Step 1: Sparse coding (OMP for each signal) ──
            for (t, signal) in signals.iter().enumerate() {
                codes[t] = omp.encode(signal, &dict)?;
            }

            // ── Step 2: Dictionary update (K-SVD per atom) ──
            for k in 0..self.n_atoms {
                // Find signals that use atom k
                let support: Vec<usize> = (0..n_signals)
                    .filter(|&t| codes[t][k].abs() > EPS)
                    .collect();
                if support.is_empty() {
                    // Re-init this atom from the signal with highest residual
                    let mut max_res = 0.0_f64;
                    let mut max_t = 0;
                    for t in 0..n_signals {
                        let residual = self.residual(signals, &dict, &codes, t);
                        let rn = norm2(&residual);
                        if rn > max_res {
                            max_res = rn;
                            max_t = t;
                        }
                    }
                    let mut new_atom = signals[max_t].clone();
                    let n = norm2(&new_atom).max(EPS);
                    for v in &mut new_atom { *v /= n; }
                    dict[k] = new_atom;
                    continue;
                }

                // Build error matrix E_k = Y - sum_{j≠k} d_j * x_j^T
                // restricted to columns in support.
                let n_sup = support.len();
                let mut e_k = vec![vec![0.0_f64; n_sup]; sig_len];
                for (col, &t) in support.iter().enumerate() {
                    for row in 0..sig_len {
                        let mut val = signals[t][row];
                        for j in 0..self.n_atoms {
                            if j == k { continue; }
                            val -= dict[j][row] * codes[t][j];
                        }
                        e_k[row][col] = val;
                    }
                }

                // Rank-1 SVD of E_k via power iteration
                let (u1, sigma1, v1) = rank1_svd(&e_k, sig_len, n_sup);

                // Update atom k and coefficients
                for row in 0..sig_len {
                    dict[k][row] = u1[row];
                }
                for (col, &t) in support.iter().enumerate() {
                    codes[t][k] = sigma1 * v1[col];
                }
            }
        }

        Ok((dict, codes))
    }

    /// Residual for signal t: y_t - D x_t.
    fn residual(
        &self,
        signals: &[Vec<f64>],
        dict: &[Vec<f64>],
        codes: &[Vec<f64>],
        t: usize,
    ) -> Vec<f64> {
        let sig_len = signals[t].len();
        let mut res = signals[t].clone();
        for k in 0..self.n_atoms {
            let c = codes[t][k];
            if c.abs() < EPS { continue; }
            for row in 0..sig_len {
                res[row] -= c * dict[k][row];
            }
        }
        res
    }

    /// Initialise dictionary from random signals (with LCG seed).
    fn init_dict(&self, signals: &[Vec<f64>], sig_len: usize) -> Vec<Vec<f64>> {
        let n_signals = signals.len();
        let mut dict = Vec::with_capacity(self.n_atoms);
        for k in 0..self.n_atoms {
            // Pick a pseudo-random signal using LCG
            let idx = (k * 7919 + 1231) % n_signals;
            let mut atom = signals[idx].clone();
            // Add a tiny perturbation to avoid degenerate initialisation
            for (i, v) in atom.iter_mut().enumerate() {
                *v += 1e-4 * ((k * sig_len + i) as f64).sin();
            }
            let n = norm2(&atom).max(EPS);
            for v in &mut atom { *v /= n; }
            dict.push(atom);
        }
        dict
    }
}

/// Rank-1 SVD via power iteration: returns (u, sigma, v).
fn rank1_svd(
    a: &[Vec<f64>],   // (rows × cols)
    rows: usize,
    cols: usize,
) -> (Vec<f64>, f64, Vec<f64>) {
    if rows == 0 || cols == 0 {
        return (vec![0.0; rows], 0.0, vec![0.0; cols]);
    }
    // Initialise v as uniform
    let mut v = vec![1.0 / (cols as f64).sqrt(); cols];
    let mut u = vec![0.0_f64; rows];

    for _ in 0..50 {
        // u = A v
        for i in 0..rows {
            u[i] = a[i].iter().zip(v.iter()).map(|(&ai, &vi)| ai * vi).sum();
        }
        let nu = norm2(&u).max(EPS);
        for val in &mut u { *val /= nu; }

        // v = A^T u
        let mut v_new = vec![0.0_f64; cols];
        for i in 0..rows {
            for j in 0..cols {
                v_new[j] += a[i][j] * u[i];
            }
        }
        let nv = norm2(&v_new).max(EPS);
        for val in &mut v_new { *val /= nv; }
        v = v_new;
    }

    // Final u = A v, sigma = ||u||
    for i in 0..rows {
        u[i] = a[i].iter().zip(v.iter()).map(|(&ai, &vi)| ai * vi).sum();
    }
    let sigma = norm2(&u).max(EPS);
    for val in &mut u { *val /= sigma; }

    (u, sigma, v)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small DCT dictionary (atoms = cosine basis vectors).
    fn dct_dict(n: usize, n_atoms: usize) -> Vec<Vec<f64>> {
        let mut dict = Vec::with_capacity(n_atoms);
        for k in 0..n_atoms {
            let atom: Vec<f64> = (0..n)
                .map(|i| ((2 * i + 1) as f64 * k as f64 * std::f64::consts::PI / (2.0 * n as f64)).cos())
                .collect();
            let norm: f64 = atom.iter().map(|&v| v * v).sum::<f64>().sqrt().max(EPS);
            dict.push(atom.iter().map(|&v| v / norm).collect())
        }
        dict
    }

    #[test]
    fn test_soft_threshold() {
        let v = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let st = soft_threshold(&v, 2.0);
        assert!((st[0] - (-1.0)).abs() < 1e-10, "st[0]={}", st[0]);
        assert_eq!(st[1], 0.0);
        assert_eq!(st[2], 0.0);
        assert_eq!(st[3], 0.0);
        assert!((st[4] - 1.0).abs() < 1e-10, "st[4]={}", st[4]);
    }

    #[test]
    fn test_hard_threshold_keeps_k_largest() {
        let v = vec![1.0, -5.0, 2.0, -3.0, 4.0];
        let ht = hard_threshold(&v, 2);
        let nonzero: Vec<usize> = ht.iter().enumerate().filter(|&(_, &x)| x.abs() > 1e-10).map(|(i, _)| i).collect();
        assert_eq!(nonzero.len(), 2, "should keep 2, got {nonzero:?}");
        // Largest two: -5 and 4
        assert!(ht[1].abs() > 1e-10 && ht[4].abs() > 1e-10);
    }

    #[test]
    fn test_omp_identity_dict() {
        // Dictionary = identity, so each signal is already sparse
        let n = 8_usize;
        let dict: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let signal: Vec<f64> = vec![0.0, 0.0, 3.0, 0.0, -2.0, 0.0, 0.0, 0.0];
        let omp = OMP::new(2);
        let code = omp.encode(&signal, &dict).expect("OMP encode");
        assert_eq!(code.len(), n);
        assert!((code[2] - 3.0).abs() < 0.1, "code[2]={}", code[2]);
        assert!((code[4] - (-2.0)).abs() < 0.1, "code[4]={}", code[4]);
    }

    #[test]
    fn test_lasso_identity_dict() {
        let n = 8_usize;
        let dict: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let signal: Vec<f64> = vec![0.0, 0.0, 2.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let lasso = Lasso::new(0.01, 500, 1e-6);
        let code = lasso.encode(&signal, &dict).expect("Lasso encode");
        assert_eq!(code.len(), n);
        assert!(code[2] > 0.5, "code[2]={}", code[2]);
        assert!(code[4] < -0.3, "code[4]={}", code[4]);
    }

    #[test]
    fn test_dict_learning_shapes() {
        let dict_init = dct_dict(16, 24);
        // Generate signals as sparse combinations of DCT atoms
        let n_signals = 40;
        let sig_len = 16;
        let signals: Vec<Vec<f64>> = (0..n_signals)
            .map(|i| {
                let a1 = &dict_init[i % 8];
                let a2 = &dict_init[(i * 3 + 1) % 12];
                a1.iter().zip(a2.iter()).map(|(&a, &b)| a + 0.5 * b).collect()
            })
            .collect();

        let dl = DictionaryLearning::new(12, 2, 10);
        let (dict, codes) = dl.fit_transform(&signals).expect("dict learning");
        assert_eq!(dict.len(), 12);
        assert_eq!(dict[0].len(), sig_len);
        assert_eq!(codes.len(), n_signals);
        assert_eq!(codes[0].len(), 12);
    }

    #[test]
    fn test_omp_dct_recovery() {
        // Signal is sum of 2 DCT atoms — OMP should recover it well
        let n = 16_usize;
        let dict = dct_dict(n, n);
        let signal: Vec<f64> = dict[3].iter().zip(dict[7].iter()).map(|(&a, &b)| 2.0 * a + 1.5 * b).collect();
        let omp = OMP::new(2);
        let code = omp.encode(&signal, &dict).expect("OMP");
        // Reconstruct
        let recon: Vec<f64> = (0..n)
            .map(|i| dict.iter().zip(code.iter()).map(|(atom, &c)| c * atom[i]).sum::<f64>())
            .collect();
        let err: f64 = recon.iter().zip(signal.iter()).map(|(&r, &s)| (r - s).powi(2)).sum::<f64>().sqrt();
        let sig_norm: f64 = signal.iter().map(|&s| s * s).sum::<f64>().sqrt();
        assert!(err / sig_norm < 0.05, "OMP reconstruction error {:.4}", err / sig_norm);
    }

    #[test]
    fn test_omp_empty_dict_error() {
        let omp = OMP::new(2);
        assert!(omp.encode(&[1.0, 2.0], &[]).is_err());
    }

    #[test]
    fn test_lasso_dimension_mismatch() {
        let dict = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let signal = vec![1.0, 2.0, 3.0]; // wrong length
        let lasso = Lasso::new(0.1, 100, 1e-6);
        assert!(lasso.encode(&signal, &dict).is_err());
    }
}
