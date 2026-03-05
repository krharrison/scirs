//! Blind Source Separation (BSS) algorithms.
//!
//! Provides JADE ICA, InfoMax ICA, SOBI, and PCA whitening.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Centre a 2-D matrix (rows = channels, cols = samples) in-place.
fn center_rows(x: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let means = x.mean_axis(Axis(1)).unwrap_or_else(|| Array1::zeros(x.nrows()));
    let mut c = x.clone();
    for i in 0..c.nrows() {
        for j in 0..c.ncols() {
            c[[i, j]] -= means[i];
        }
    }
    (c, means)
}

/// Covariance matrix C = X X^T / (n-1).
fn cov(x: &Array2<f64>) -> Array2<f64> {
    let n = x.ncols().max(1) as f64;
    x.dot(&x.t()) / (n - 1.0).max(1.0)
}

/// Symmetric square-root inverse  C^{-1/2}  via eigendecomposition.
/// Returns (W_whiten, eigenvalues) where W_whiten is the whitening matrix.
fn sym_sqrt_inv(c: &Array2<f64>) -> SignalResult<(Array2<f64>, Array1<f64>)> {
    let p = c.nrows();
    // Power iteration based symmetric eigendecomposition (Jacobi method)
    let (eigvals, eigvecs) = jacobi_eigh(c)?;

    // Build D^{-1/2}
    let mut d_inv_sqrt = Array1::<f64>::zeros(p);
    for i in 0..p {
        let v = eigvals[i];
        if v > 1e-12 {
            d_inv_sqrt[i] = 1.0 / v.sqrt();
        } else {
            d_inv_sqrt[i] = 0.0;
        }
    }

    // W = V * D^{-1/2} * V^T
    let mut w = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..p {
                s += eigvecs[[i, k]] * d_inv_sqrt[k] * eigvecs[[j, k]];
            }
            w[[i, j]] = s;
        }
    }
    Ok((w, eigvals))
}

/// Jacobi symmetric eigendecomposition. Returns (eigenvalues, eigenvectors).
/// Eigenvectors are columns of the returned matrix.
fn jacobi_eigh(a: &Array2<f64>) -> SignalResult<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SignalError::DimensionMismatch(
            "Jacobi eigh requires square matrix".into(),
        ));
    }
    let mut mat = a.clone();
    let mut v = Array2::<f64>::eye(n);

    for _sweep in 0..200 {
        // Find off-diagonal element with largest magnitude
        let mut max_val = 0.0_f64;
        let mut p_idx = 0;
        let mut q_idx = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = mat[[i, j]].abs();
                if val > max_val {
                    max_val = val;
                    p_idx = i;
                    q_idx = j;
                }
            }
        }
        if max_val < 1e-15 {
            break;
        }
        // Compute Jacobi rotation
        let tau = (mat[[q_idx, q_idx]] - mat[[p_idx, p_idx]]) / (2.0 * mat[[p_idx, q_idx]]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            1.0 / (tau - (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Update mat
        let app = mat[[p_idx, p_idx]];
        let aqq = mat[[q_idx, q_idx]];
        let apq = mat[[p_idx, q_idx]];
        mat[[p_idx, p_idx]] = app - t * apq;
        mat[[q_idx, q_idx]] = aqq + t * apq;
        mat[[p_idx, q_idx]] = 0.0;
        mat[[q_idx, p_idx]] = 0.0;
        for r in 0..n {
            if r != p_idx && r != q_idx {
                let arp = mat[[r, p_idx]];
                let arq = mat[[r, q_idx]];
                mat[[r, p_idx]] = c * arp - s * arq;
                mat[[p_idx, r]] = mat[[r, p_idx]];
                mat[[r, q_idx]] = s * arp + c * arq;
                mat[[q_idx, r]] = mat[[r, q_idx]];
            }
        }
        // Update eigenvectors
        for r in 0..n {
            let vrp = v[[r, p_idx]];
            let vrq = v[[r, q_idx]];
            v[[r, p_idx]] = c * vrp - s * vrq;
            v[[r, q_idx]] = s * vrp + c * vrq;
        }
    }

    let eigvals = Array1::from_iter((0..n).map(|i| mat[[i, i]]));
    Ok((eigvals, v))
}

/// Convert `Vec<Vec<f64>>` (rows × cols) to `Array2<f64>`.
fn vv_to_arr2(data: &[Vec<f64>]) -> SignalResult<Array2<f64>> {
    let rows = data.len();
    if rows == 0 {
        return Err(SignalError::ValueError("Empty input matrix".into()));
    }
    let cols = data[0].len();
    if cols == 0 {
        return Err(SignalError::ValueError("Empty row in input matrix".into()));
    }
    let flat: Vec<f64> = data.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| SignalError::ShapeMismatch(e.to_string()))
}

/// Convert `Array2<f64>` to `Vec<Vec<f64>>`.
fn arr2_to_vv(a: &Array2<f64>) -> Vec<Vec<f64>> {
    (0..a.nrows())
        .map(|i| a.row(i).to_vec())
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// PCA whitening.
///
/// Returns `(whitened, W_whiten, components)` where:
/// - `whitened` has shape `(n_components, n_samples)`,
/// - `W_whiten` has shape `(n_components, n_channels)`,
/// - `components` has shape `(n_components, n_channels)` (top eigenvectors).
pub fn pca_whitening(
    x: &[Vec<f64>],
    n_components: usize,
) -> SignalResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let mat = vv_to_arr2(x)?;
    let (centered, _means) = center_rows(&mat);
    let c = cov(&centered);

    let (eigvals, eigvecs) = jacobi_eigh(&c)?;
    let p = c.nrows();

    // Sort indices by descending eigenvalue
    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_by(|&a, &b| eigvals[b].partial_cmp(&eigvals[a]).unwrap_or(std::cmp::Ordering::Equal));

    let k = n_components.min(p);

    // Build whitening matrix W (k × p): W_ij = V_ij / sqrt(lambda_j)
    let mut w_whiten = Array2::<f64>::zeros((k, p));
    let mut components = Array2::<f64>::zeros((k, p));
    for (row, &ci) in idx[..k].iter().enumerate() {
        let scale = if eigvals[ci] > 1e-12 {
            1.0 / eigvals[ci].sqrt()
        } else {
            0.0
        };
        for col in 0..p {
            w_whiten[[row, col]] = eigvecs[[col, ci]] * scale;
            components[[row, col]] = eigvecs[[col, ci]];
        }
    }

    let whitened = w_whiten.dot(&centered);
    Ok((arr2_to_vv(&whitened), arr2_to_vv(&w_whiten), arr2_to_vv(&components)))
}

// ──────────────────────────────────────────────────────────────────────────────
// JADE ICA
// ──────────────────────────────────────────────────────────────────────────────

/// JADE (Joint Approximate Diagonalization of Eigenmatrices) ICA.
///
/// Steps:
/// 1. Whiten the data via PCA.
/// 2. Compute a set of normalised 4th-order cumulant matrices.
/// 3. Perform Jacobi joint-diagonalisation sweeps.
///
/// # Returns
/// `(unmixing_matrix, sources)` where each row is a source signal.
pub fn jade_ica(
    x: &[Vec<f64>],
    n_sources: usize,
) -> SignalResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let mat = vv_to_arr2(x)?;
    let (n_channels, n_samples) = mat.dim();
    if n_sources == 0 || n_sources > n_channels {
        return Err(SignalError::InvalidArgument(format!(
            "n_sources must be in 1..={n_channels}"
        )));
    }
    if n_samples < 2 {
        return Err(SignalError::InvalidArgument(
            "Need at least 2 samples".into(),
        ));
    }

    // Step 1: whiten
    let (whitened_vv, w_whiten_vv, _) = pca_whitening(x, n_sources)?;
    let z = vv_to_arr2(&whitened_vv)?;   // (n_sources × n_samples)
    let w_whiten = vv_to_arr2(&w_whiten_vv)?;   // (n_sources × n_channels)

    let k = n_sources;

    // Step 2: build cumulant matrices
    // For each pair (p,q) in the whitened space, compute C_{pq} = E[z_p z_q z z^T] - ...
    // We use the simplified normalised cumulant matrices from the JADE paper.
    let mut cum_mats: Vec<Array2<f64>> = Vec::new();

    for p in 0..k {
        for q in p..k {
            // C_pq[i,j] = (1/N) sum_t z_p[t]*z_q[t]*z_i[t]*z_j[t]
            //              - delta(i,p)*Czq_j  - delta(j,p)*Czq_i
            //              - delta(i,q)*Czp_j  - delta(j,q)*Czp_i
            // where Czp_i = (1/N) sum_t z_p[t]*z_i[t]  (covariance, =delta(p,i) for white data)
            let mut cm = Array2::<f64>::zeros((k, k));
            for i in 0..k {
                for j in 0..k {
                    let mut s = 0.0_f64;
                    for t in 0..n_samples {
                        s += z[[p, t]] * z[[q, t]] * z[[i, t]] * z[[j, t]];
                    }
                    cm[[i, j]] = s / n_samples as f64;
                }
            }
            // Subtract Gaussian contributions (whitened data: cov = I)
            for i in 0..k {
                cm[[i, i]] -= if i == p { 1.0 } else { 0.0 }; // delta(i,p)*delta(q,i)
                cm[[i, i]] -= if i == q { 1.0 } else { 0.0 }; // delta(i,q)*delta(p,i)
            }
            // Symmetrize
            for i in 0..k {
                for j in 0..k {
                    let v = (cm[[i, j]] + cm[[j, i]]) * 0.5;
                    cm[[i, j]] = v;
                    cm[[j, i]] = v;
                }
            }
            cum_mats.push(cm);
        }
    }

    // Step 3: Jacobi joint diagonalisation
    // Initialise rotation as identity
    let mut rot = Array2::<f64>::eye(k);

    for _sweep in 0..100 {
        let mut max_off = 0.0_f64;
        for p in 0..k {
            for q in (p + 1)..k {
                // Compute Givens rotation angle theta that minimises sum ||off-diag||^2
                // across all cumulant matrices for the (p,q) pair.
                let mut g11 = 0.0_f64;
                let mut g22 = 0.0_f64;
                let mut g12 = 0.0_f64;
                for cm in &cum_mats {
                    let app = cm[[p, p]];
                    let aqq = cm[[q, q]];
                    let apq = cm[[p, q]];
                    g11 += app * app - aqq * aqq;
                    g22 += 2.0 * apq * (app + aqq);
                    g12 += 2.0 * apq * apq - (app - aqq).powi(2) / 2.0;
                }
                max_off = max_off.max(g12.abs());
                let theta = 0.5 * (g22 / g11.hypot(g22)).asin();
                if theta.abs() < 1e-15 {
                    continue;
                }
                let c = theta.cos();
                let s = theta.sin();
                // Apply rotation to all cumulant matrices
                for cm in &mut cum_mats {
                    // Rotate rows p and q
                    for j in 0..k {
                        let rp = cm[[p, j]];
                        let rq = cm[[q, j]];
                        cm[[p, j]] = c * rp + s * rq;
                        cm[[q, j]] = -s * rp + c * rq;
                    }
                    // Rotate cols p and q
                    for i in 0..k {
                        let rp = cm[[i, p]];
                        let rq = cm[[i, q]];
                        cm[[i, p]] = c * rp + s * rq;
                        cm[[i, q]] = -s * rp + c * rq;
                    }
                }
                // Accumulate rotation
                for i in 0..k {
                    let rp = rot[[i, p]];
                    let rq = rot[[i, q]];
                    rot[[i, p]] = c * rp + s * rq;
                    rot[[i, q]] = -s * rp + c * rq;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }
    }

    // Unmixing = rot^T * W_whiten  (shape k × n_channels)
    let unmixing = rot.t().dot(&w_whiten);
    let sources = unmixing.dot(&mat);

    Ok((arr2_to_vv(&unmixing), arr2_to_vv(&sources)))
}

// ──────────────────────────────────────────────────────────────────────────────
// InfoMax ICA (Bell-Sejnowski)
// ──────────────────────────────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// InfoMax ICA (Bell-Sejnowski 1995).
///
/// Score function `g(u) = 1 - 2·sigmoid(u)` (assumes super-Gaussian sources by default).
///
/// Update rule: `W += lr * (I + g(y)·y^T) * W`
///
/// # Returns
/// `(unmixing_matrix, sources)`
pub fn infomax_ica(
    x: &[Vec<f64>],
    n_sources: usize,
    max_iter: usize,
    lr: f64,
) -> SignalResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let mat = vv_to_arr2(x)?;
    let (n_channels, n_samples) = mat.dim();
    if n_sources == 0 || n_sources > n_channels {
        return Err(SignalError::InvalidArgument(format!(
            "n_sources must be in 1..={n_channels}"
        )));
    }
    if n_samples < 2 {
        return Err(SignalError::InvalidArgument(
            "Need at least 2 samples".into(),
        ));
    }

    // Whiten first
    let (whitened_vv, w_whiten_vv, _) = pca_whitening(x, n_sources)?;
    let z = vv_to_arr2(&whitened_vv)?;
    let w_whiten = vv_to_arr2(&w_whiten_vv)?;

    let k = n_sources;
    let mut w = Array2::<f64>::eye(k);

    for _iter in 0..max_iter {
        // y = W * z  (k × n_samples)
        let y = w.dot(&z);

        // Compute gradient update summed over samples
        // dW = (I + g(y)*y^T) averaged over t, then * W
        // g(y_t) has shape k; y_t has shape k => outer product k×k
        let mut outer_sum = Array2::<f64>::zeros((k, k));
        for t in 0..n_samples {
            for i in 0..k {
                let gy = 1.0 - 2.0 * sigmoid(y[[i, t]]);
                for j in 0..k {
                    outer_sum[[i, j]] += gy * y[[j, t]];
                }
            }
        }
        outer_sum.mapv_inplace(|v| v / n_samples as f64);

        // dW = (I + outer_sum) * W
        let identity = Array2::<f64>::eye(k);
        let dw = (&identity + &outer_sum).dot(&w);
        w = &w + &dw.mapv(|v| v * lr);

        // Decorrelation / stability: normalise rows
        for i in 0..k {
            let norm: f64 = w.row(i).iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > 1e-12 {
                w.row_mut(i).mapv_inplace(|v| v / norm);
            }
        }
    }

    // Full unmixing: w * w_whiten  (k × n_channels)
    let unmixing = w.dot(&w_whiten);
    let sources = unmixing.dot(&mat);

    Ok((arr2_to_vv(&unmixing), arr2_to_vv(&sources)))
}

// ──────────────────────────────────────────────────────────────────────────────
// SOBI
// ──────────────────────────────────────────────────────────────────────────────

/// SOBI: Second Order Blind Identification.
///
/// Simultaneously diagonalises correlation matrices at multiple lags.
///
/// # Arguments
/// * `x`    — rows = channels, cols = samples
/// * `lags` — time lags to use (e.g. `&[1, 2, 3, 4]`)
///
/// # Returns
/// `(unmixing_matrix, sources)`
pub fn sobi(
    x: &[Vec<f64>],
    lags: &[usize],
) -> SignalResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let mat = vv_to_arr2(x)?;
    let (n_channels, n_samples) = mat.dim();
    if n_samples < 2 {
        return Err(SignalError::InvalidArgument(
            "Need at least 2 samples for SOBI".into(),
        ));
    }
    if lags.is_empty() {
        return Err(SignalError::InvalidArgument(
            "At least one lag required".into(),
        ));
    }

    // Step 1: whiten
    let k = n_channels;
    let (whitened_vv, w_whiten_vv, _) = pca_whitening(x, k)?;
    let z = vv_to_arr2(&whitened_vv)?;
    let w_whiten = vv_to_arr2(&w_whiten_vv)?;

    // Step 2: compute lagged correlation matrices  R(tau) = (1/N) * Z[:,tau:] * Z[:,:-tau]^T
    let mut corr_mats: Vec<Array2<f64>> = Vec::new();
    for &lag in lags {
        if lag == 0 || lag >= n_samples {
            continue;
        }
        let t = n_samples - lag;
        let mut r = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0_f64;
                for ti in 0..t {
                    s += z[[i, ti + lag]] * z[[j, ti]];
                }
                r[[i, j]] = s / t as f64;
            }
        }
        // Symmetrize
        let r_sym = (&r + &r.t()) * 0.5;
        corr_mats.push(r_sym);
    }

    if corr_mats.is_empty() {
        return Err(SignalError::InvalidArgument(
            "No valid lags produced correlation matrices".into(),
        ));
    }

    // Step 3: Joint diagonalisation (same Jacobi sweeps as JADE)
    let mut v = Array2::<f64>::eye(k);

    for _sweep in 0..200 {
        let mut max_off = 0.0_f64;
        for p in 0..k {
            for q in (p + 1)..k {
                let mut g11 = 0.0_f64;
                let mut g22 = 0.0_f64;
                for cm in &corr_mats {
                    let app = cm[[p, p]];
                    let aqq = cm[[q, q]];
                    let apq = cm[[p, q]];
                    g11 += app - aqq;
                    g22 += 2.0 * apq;
                    max_off = max_off.max(apq.abs());
                }
                let denom = g11.hypot(g22);
                if denom < 1e-15 {
                    continue;
                }
                let theta = 0.5 * (g22 / denom).asin();
                let c = theta.cos();
                let s = theta.sin();
                for cm in &mut corr_mats {
                    for j in 0..k {
                        let rp = cm[[p, j]];
                        let rq = cm[[q, j]];
                        cm[[p, j]] = c * rp + s * rq;
                        cm[[q, j]] = -s * rp + c * rq;
                    }
                    for i in 0..k {
                        let rp = cm[[i, p]];
                        let rq = cm[[i, q]];
                        cm[[i, p]] = c * rp + s * rq;
                        cm[[i, q]] = -s * rp + c * rq;
                    }
                }
                for i in 0..k {
                    let rp = v[[i, p]];
                    let rq = v[[i, q]];
                    v[[i, p]] = c * rp + s * rq;
                    v[[i, q]] = -s * rp + c * rq;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }
    }

    // Unmixing matrix: V^T * W_whiten
    let unmixing = v.t().dot(&w_whiten);
    let sources = unmixing.dot(&mat);

    Ok((arr2_to_vv(&unmixing), arr2_to_vv(&sources)))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_mix() -> Vec<Vec<f64>> {
        // Two linearly independent sources mixed into two channels
        let n = 64_usize;
        let s1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let s2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        let x1: Vec<f64> = s1.iter().zip(s2.iter()).map(|(&a, &b)| 0.8 * a + 0.2 * b).collect();
        let x2: Vec<f64> = s1.iter().zip(s2.iter()).map(|(&a, &b)| 0.3 * a + 0.7 * b).collect();
        vec![x1, x2]
    }

    #[test]
    fn test_pca_whitening_shapes() {
        let x = synthetic_mix();
        let (wh, wmat, comp) = pca_whitening(&x, 2).expect("pca_whitening should succeed");
        assert_eq!(wh.len(), 2);
        assert_eq!(wmat.len(), 2);
        assert_eq!(comp.len(), 2);
        assert_eq!(wh[0].len(), x[0].len());
    }

    #[test]
    fn test_jade_ica_shapes() {
        let x = synthetic_mix();
        let (unmixing, sources) = jade_ica(&x, 2).expect("jade_ica should succeed");
        assert_eq!(unmixing.len(), 2);
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].len(), x[0].len());
    }

    #[test]
    fn test_infomax_ica_shapes() {
        let x = synthetic_mix();
        let (unmixing, sources) = infomax_ica(&x, 2, 50, 0.01).expect("infomax_ica should succeed");
        assert_eq!(unmixing.len(), 2);
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].len(), x[0].len());
    }

    #[test]
    fn test_sobi_shapes() {
        let x = synthetic_mix();
        let (unmixing, sources) = sobi(&x, &[1, 2, 3]).expect("sobi should succeed");
        assert_eq!(unmixing.len(), 2);
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0].len(), x[0].len());
    }

    #[test]
    fn test_pca_whitening_uncorrelated() {
        let x = synthetic_mix();
        let (whitened, _, _) = pca_whitening(&x, 2).expect("pca_whitening");
        let wmat = vv_to_arr2(&whitened).expect("arr");
        // Covariance of whitened data should be near identity
        let c = cov(&wmat);
        for i in 0..2 {
            assert!((c[[i, i]] - 1.0).abs() < 0.3, "diagonal c[{i},{i}]={}", c[[i, i]]);
            for j in 0..2 {
                if i != j {
                    assert!(c[[i, j]].abs() < 0.3, "off-diag c[{i},{j}]={}", c[[i, j]]);
                }
            }
        }
    }

    #[test]
    fn test_jade_invalid_n_sources() {
        let x = synthetic_mix();
        assert!(jade_ica(&x, 0).is_err());
        assert!(jade_ica(&x, 10).is_err());
    }

    #[test]
    fn test_infomax_invalid() {
        let x = synthetic_mix();
        assert!(infomax_ica(&x, 0, 10, 0.01).is_err());
    }

    #[test]
    fn test_sobi_empty_lags() {
        let x = synthetic_mix();
        assert!(sobi(&x, &[]).is_err());
    }
}
