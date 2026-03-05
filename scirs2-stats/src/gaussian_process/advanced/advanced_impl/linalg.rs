//! Internal linear-algebra helpers for advanced GP methods.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Lower-triangular Cholesky: A = L Lᵀ.
pub(crate) fn cholesky(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0_f64;
            for k in 0..j {
                s += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let v = a[[i, i]] - s;
                if v < 0.0 {
                    return Err(StatsError::ComputationError(format!(
                        "Matrix not positive definite at diagonal {i} (value={v:.3e})"
                    )));
                }
                l[[i, j]] = v.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - s) / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Cholesky with automatic jitter fallback.
pub(crate) fn cholesky_jitter(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    match cholesky(a) {
        Ok(l) => Ok(l),
        Err(_) => {
            let n = a.nrows();
            let mut aa = a.clone();
            for jitter in [1e-8, 1e-6, 1e-4, 1e-2] {
                for i in 0..n {
                    aa[[i, i]] += jitter;
                }
                if let Ok(l) = cholesky(&aa) {
                    return Ok(l);
                }
            }
            Err(StatsError::ComputationError(
                "Cholesky failed even with maximum jitter".into(),
            ))
        }
    }
}

/// Forward substitution: solve L x = b.
pub(crate) fn solve_lower(l: &Array2<f64>, b: &Array1<f64>) -> StatsResult<Array1<f64>> {
    let n = l.nrows();
    let mut x = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..i {
            s += l[[i, j]] * x[j];
        }
        let d = l[[i, i]];
        if d.abs() < 1e-14 {
            return Err(StatsError::ComputationError(
                "Singular lower-triangular matrix".into(),
            ));
        }
        x[i] = (b[i] - s) / d;
    }
    Ok(x)
}

/// Back substitution: solve Uᵀ x = b (U is upper triangular).
pub(crate) fn solve_upper(u: &Array2<f64>, b: &Array1<f64>) -> StatsResult<Array1<f64>> {
    let n = u.nrows();
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = 0.0_f64;
        for j in (i + 1)..n {
            s += u[[i, j]] * x[j];
        }
        let d = u[[i, i]];
        if d.abs() < 1e-14 {
            return Err(StatsError::ComputationError(
                "Singular upper-triangular matrix".into(),
            ));
        }
        x[i] = (b[i] - s) / d;
    }
    Ok(x)
}

/// Solve A x = b via Cholesky (A symmetric PD).
pub(crate) fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> StatsResult<Array1<f64>> {
    let l = cholesky_jitter(a)?;
    let y = solve_lower(&l, b)?;
    solve_upper(&l.t().to_owned(), &y)
}

/// Solve A X = B column by column.
pub(crate) fn solve_spd_matrix(a: &Array2<f64>, b: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let l = cholesky_jitter(a)?;
    let ncols = b.ncols();
    let mut x = Array2::<f64>::zeros((a.nrows(), ncols));
    for c in 0..ncols {
        let bc = b.column(c).to_owned();
        let y = solve_lower(&l, &bc)?;
        let xc = solve_upper(&l.t().to_owned(), &y)?;
        for r in 0..a.nrows() {
            x[[r, c]] = xc[r];
        }
    }
    Ok(x)
}

/// Solve L X = B (column by column).
pub(crate) fn solve_lower_matrix(l: &Array2<f64>, b: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let ncols = b.ncols();
    let mut x = Array2::<f64>::zeros((l.nrows(), ncols));
    for c in 0..ncols {
        let bc = b.column(c).to_owned();
        let xc = solve_lower(l, &bc)?;
        for r in 0..l.nrows() {
            x[[r, c]] = xc[r];
        }
    }
    Ok(x)
}

/// log|A| = 2 Σ log diag(L).
pub(crate) fn log_det_from_cholesky(l: &Array2<f64>) -> f64 {
    2.0 * l.diag().iter().map(|&v| v.ln()).sum::<f64>()
}
