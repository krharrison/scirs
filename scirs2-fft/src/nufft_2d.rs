//! 2-D Non-Uniform Fast Fourier Transform (NUFFT-2D).
//!
//! Implements separable spreading / interpolation for Type-1 and Type-2
//! two-dimensional NUFFTs.
//!
//! # Algorithm
//!
//! Both types use the same three-step approach as the 1-D case, but applied on a
//! 2-D oversampled Cartesian grid:
//!
//! 1. **Spread** (Type-1) or **place** (Type-2): Map between the non-uniform
//!    points and the oversampled grid using a separable Gaussian kernel.
//! 2. **FFT**: Apply a standard 2-D FFT (row-column decomposition).
//! 3. **Deconvolve**: Correct for the kernel attenuation by multiplying each
//!    output mode by the appropriate correction factor.
//!
//! The separability of the Gaussian kernel means the spreading and interpolation
//! factorises as a product of two 1-D operations, avoiding the need for a
//! full 2-D convolution.
//!
//! # References
//!
//! * Dutt, A., & Rokhlin, V. (1993). Fast Fourier transforms for nonequispaced data.
//!   SIAM Journal on Scientific Computing, 14(6), 1368-1393.
//! * Greengard, L., & Lee, J. Y. (2004). Accelerating the nonuniform fast Fourier transform.
//!   SIAM Review, 46(3), 443-464.

use crate::error::{FFTError, FFTResult};
use crate::nufft::{fft_internal, gaussian_correction, gaussian_kernel, ifft_internal, OVERSAMPLE};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::Zero;
use std::f64::consts::PI;

// ─── Type-1 2-D NUFFT ────────────────────────────────────────────────────────

/// Type-1 2-D NUFFT: non-uniform sources → uniform 2-D Fourier grid.
///
/// Computes
/// ```text
/// F̂[k₁, k₂] = Σ_j c_j · exp(−i · (k₁·x_j + k₂·y_j))
/// ```
/// for `(k₁, k₂) ∈ {−N₁/2,…,N₁/2−1} × {−N₂/2,…,N₂/2−1}`.
///
/// Uses a separable Gaussian spreading kernel with oversampling factor σ = 2.
///
/// # Arguments
///
/// * `xy`       – non-uniform source points as `&[(x, y)]`, each coordinate in `[-π, π)`
/// * `c`        – complex strengths at each source point (same length as `xy`)
/// * `n_modes`  – output grid dimensions `(N₁, N₂)` (row × column)
/// * `eps`      – desired relative accuracy (e.g. `1e-6`)
///
/// # Returns
///
/// `Array2<Complex64>` of shape `(N₁, N₂)` where element `[i, j]` corresponds to
/// mode `(i − N₁/2, j − N₂/2)`.
///
/// # Errors
///
/// * `DimensionError` – if `xy` and `c` have different lengths
/// * `ValueError`     – if `eps ≤ 0`, `n_modes` contains a zero, or any coordinate
///   lies outside `[-π, π)`
///
/// # Examples
///
/// ```
/// use scirs2_fft::nufft_2d::nufft2_type1;
/// use scirs2_core::numeric::Complex64;
/// use std::f64::consts::PI;
///
/// let n = 16usize;
/// let xy: Vec<(f64, f64)> = (0..n)
///     .map(|j| {
///         let t = 2.0 * PI * j as f64 / n as f64 - PI;
///         (t, t * 0.5)
///     })
///     .collect();
/// let c: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); n];
///
/// let f_hat = nufft2_type1(&xy, &c, (8, 8), 1e-6).expect("nufft2_type1");
/// assert_eq!(f_hat.shape(), &[8, 8]);
/// ```
pub fn nufft2_type1(
    xy: &[(f64, f64)],
    c: &[Complex64],
    n_modes: (usize, usize),
    eps: f64,
) -> FFTResult<Array2<Complex64>> {
    validate_inputs_2d(xy, c, n_modes, eps)?;

    let (n1, n2) = n_modes;
    let sigma = OVERSAMPLE;

    let ng1 = oversample_grid_size(sigma, n1);
    let ng2 = oversample_grid_size(sigma, n2);
    let half_w = kernel_half_width_2d(sigma, eps);

    // Step 1: Spread onto oversampled 2-D grid (row-major, shape [ng1, ng2])
    let mut grid = vec![Complex64::zero(); ng1 * ng2];
    spread_type1_2d(xy, c, &mut grid, ng1, ng2, sigma, half_w);

    // Step 2: 2-D FFT via row–column decomposition
    let grid_fft = fft2d_row_col(&grid, ng1, ng2)?;

    // Step 3: Extract modes and deconvolve
    let half1 = (n1 / 2) as i64;
    let half2 = (n2 / 2) as i64;

    let result = Array2::from_shape_fn((n1, n2), |(r, s)| {
        let k1 = r as i64 - half1;
        let k2 = s as i64 - half2;
        let bin1 = ((k1).rem_euclid(ng1 as i64)) as usize;
        let bin2 = ((k2).rem_euclid(ng2 as i64)) as usize;
        let val = grid_fft[bin1 * ng2 + bin2];
        let corr1 = gaussian_correction(k1, ng1, sigma);
        let corr2 = gaussian_correction(k2, ng2, sigma);
        val * corr1 * corr2
    });

    Ok(result)
}

// ─── Type-2 2-D NUFFT ────────────────────────────────────────────────────────

/// Type-2 2-D NUFFT: uniform 2-D Fourier data → non-uniform points.
///
/// Computes
/// ```text
/// f(x_j, y_j) = Σ_{k₁,k₂} F̂[k₁, k₂] · exp(i · (k₁·x_j + k₂·y_j))
/// ```
/// where the sums run over the centred modes stored in `f_hat`.
///
/// This is the adjoint (transpose conjugate) of [`nufft2_type1`].
///
/// # Arguments
///
/// * `f_hat` – uniform 2-D Fourier coefficients, shape `(N₁, N₂)`, centred convention
/// * `xy`    – non-uniform target points `&[(x, y)]`, each coordinate in `[-π, π)`
/// * `eps`   – desired relative accuracy
///
/// # Returns
///
/// `Vec<Complex64>` of length `M = xy.len()` containing `f(x_j, y_j)`.
///
/// # Errors
///
/// * `ValueError` – if `eps ≤ 0` or any coordinate lies outside `[-π, π)`
/// * `DimensionError` – if `f_hat` is empty
///
/// # Examples
///
/// ```
/// use scirs2_fft::nufft_2d::nufft2_type2;
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::numeric::Complex64;
/// use std::f64::consts::PI;
///
/// let n1 = 8usize;
/// let n2 = 8usize;
/// let f_hat = Array2::from_elem((n1, n2), Complex64::new(1.0, 0.0));
///
/// let m = 16usize;
/// let xy: Vec<(f64, f64)> = (0..m)
///     .map(|j| {
///         let t = 2.0 * PI * j as f64 / m as f64 - PI;
///         (t, -t)
///     })
///     .collect();
///
/// let vals = nufft2_type2(&f_hat, &xy, 1e-6).expect("nufft2_type2");
/// assert_eq!(vals.len(), m);
/// ```
pub fn nufft2_type2(
    f_hat: &Array2<Complex64>,
    xy: &[(f64, f64)],
    eps: f64,
) -> FFTResult<Vec<Complex64>> {
    let shape = f_hat.shape();
    let n1 = shape[0];
    let n2 = shape[1];

    if n1 == 0 || n2 == 0 {
        return Err(FFTError::DimensionError(
            "f_hat must have non-zero dimensions".to_string(),
        ));
    }
    if eps <= 0.0 {
        return Err(FFTError::ValueError("eps must be positive".to_string()));
    }
    for &(xj, yj) in xy {
        if !(-PI..PI).contains(&xj) || !(-PI..PI).contains(&yj) {
            return Err(FFTError::ValueError(
                "all xy coordinates must lie in [-π, π)".to_string(),
            ));
        }
    }

    let sigma = OVERSAMPLE;
    let ng1 = oversample_grid_size(sigma, n1);
    let ng2 = oversample_grid_size(sigma, n2);
    let half_w = kernel_half_width_2d(sigma, eps);

    let half1 = (n1 / 2) as i64;
    let half2 = (n2 / 2) as i64;

    // Step 1: Place (deconvolved) Fourier coefficients in the oversampled grid
    let mut grid_freq = vec![Complex64::zero(); ng1 * ng2];
    for r in 0..n1 {
        for s in 0..n2 {
            let k1 = r as i64 - half1;
            let k2 = s as i64 - half2;
            let corr1 = gaussian_correction(k1, ng1, sigma);
            let corr2 = gaussian_correction(k2, ng2, sigma);
            let bin1 = ((k1).rem_euclid(ng1 as i64)) as usize;
            let bin2 = ((k2).rem_euclid(ng2 as i64)) as usize;
            grid_freq[bin1 * ng2 + bin2] = f_hat[(r, s)] * corr1 * corr2;
        }
    }

    // Step 2: 2-D IFFT via row–column decomposition
    let grid_time = ifft2d_row_col(&grid_freq, ng1, ng2)?;

    // Step 3: Interpolate at non-uniform target points
    let out = interpolate_type2_2d(xy, &grid_time, ng1, ng2, sigma, half_w);

    Ok(out)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute the oversampled grid size for dimension `n`.
fn oversample_grid_size(sigma: f64, n: usize) -> usize {
    let raw = (sigma * n as f64).ceil() as usize;
    if raw % 2 == 0 { raw } else { raw + 1 }
}

/// Kernel half-width (in grid cells) for the 2-D Gaussian NUFFT.
fn kernel_half_width_2d(sigma: f64, eps: f64) -> usize {
    let w = sigma * ((-eps.ln()) / (PI * PI)).sqrt();
    (w.ceil() as usize).max(2)
}

/// Validate inputs common to both Type-1 and Type-2 callers.
fn validate_inputs_2d(
    xy: &[(f64, f64)],
    c: &[Complex64],
    n_modes: (usize, usize),
    eps: f64,
) -> FFTResult<()> {
    if xy.len() != c.len() {
        return Err(FFTError::DimensionError(
            "xy and c must have the same length".to_string(),
        ));
    }
    if n_modes.0 == 0 || n_modes.1 == 0 {
        return Err(FFTError::ValueError(
            "n_modes dimensions must be > 0".to_string(),
        ));
    }
    if eps <= 0.0 {
        return Err(FFTError::ValueError("eps must be positive".to_string()));
    }
    for &(xj, yj) in xy {
        if !(-PI..PI).contains(&xj) || !(-PI..PI).contains(&yj) {
            return Err(FFTError::ValueError(
                "all xy coordinates must lie in [-π, π)".to_string(),
            ));
        }
    }
    Ok(())
}

/// Spread non-uniform 2-D sources onto the oversampled grid (Type-1).
///
/// Uses the separability of the Gaussian kernel:
/// ```text
/// w(x, y) = w₁(x) · w₂(y)
/// ```
fn spread_type1_2d(
    xy: &[(f64, f64)],
    c: &[Complex64],
    grid: &mut [Complex64],
    ng1: usize,
    ng2: usize,
    sigma: f64,
    half_w: usize,
) {
    let h1 = 2.0 * PI / ng1 as f64;
    let h2 = 2.0 * PI / ng2 as f64;
    let half_w_i = half_w as isize;

    for (&(xj, yj), &cj) in xy.iter().zip(c.iter()) {
        // Map onto grid coordinates
        let xg = (xj + PI) / h1;
        let yg = (yj + PI) / h2;
        let ix0 = xg.floor() as isize;
        let iy0 = yg.floor() as isize;

        // Pre-compute 1-D kernel weights for x direction
        let wx: Vec<f64> = ((-half_w_i)..=(half_w_i))
            .map(|di| gaussian_kernel(xg - (ix0 + di) as f64, sigma))
            .collect();

        // Pre-compute 1-D kernel weights for y direction
        let wy: Vec<f64> = ((-half_w_i)..=(half_w_i))
            .map(|dj| gaussian_kernel(yg - (iy0 + dj) as f64, sigma))
            .collect();

        for (di_idx, di) in ((-half_w_i)..=(half_w_i)).enumerate() {
            let ridx = (ix0 + di).rem_euclid(ng1 as isize) as usize;
            let wxd = wx[di_idx];
            for (dj_idx, dj) in ((-half_w_i)..=(half_w_i)).enumerate() {
                let cidx = (iy0 + dj).rem_euclid(ng2 as isize) as usize;
                let w = wxd * wy[dj_idx];
                grid[ridx * ng2 + cidx] += cj * w;
            }
        }
    }
}

/// Interpolate values at non-uniform 2-D target points from the oversampled grid (Type-2).
fn interpolate_type2_2d(
    xy: &[(f64, f64)],
    grid: &[Complex64],
    ng1: usize,
    ng2: usize,
    sigma: f64,
    half_w: usize,
) -> Vec<Complex64> {
    let h1 = 2.0 * PI / ng1 as f64;
    let h2 = 2.0 * PI / ng2 as f64;
    let half_w_i = half_w as isize;
    let mut out = vec![Complex64::zero(); xy.len()];

    for (out_j, &(xj, yj)) in out.iter_mut().zip(xy.iter()) {
        let xg = (xj + PI) / h1;
        let yg = (yj + PI) / h2;
        let ix0 = xg.floor() as isize;
        let iy0 = yg.floor() as isize;

        let wx: Vec<f64> = ((-half_w_i)..=(half_w_i))
            .map(|di| gaussian_kernel(xg - (ix0 + di) as f64, sigma))
            .collect();
        let wy: Vec<f64> = ((-half_w_i)..=(half_w_i))
            .map(|dj| gaussian_kernel(yg - (iy0 + dj) as f64, sigma))
            .collect();

        let mut acc = Complex64::zero();
        for (di_idx, di) in ((-half_w_i)..=(half_w_i)).enumerate() {
            let ridx = (ix0 + di).rem_euclid(ng1 as isize) as usize;
            let wxd = wx[di_idx];
            for (dj_idx, dj) in ((-half_w_i)..=(half_w_i)).enumerate() {
                let cidx = (iy0 + dj).rem_euclid(ng2 as isize) as usize;
                acc += grid[ridx * ng2 + cidx] * (wxd * wy[dj_idx]);
            }
        }
        *out_j = acc;
    }
    out
}

/// Row-column 2-D FFT (in-place on flattened row-major buffer).
fn fft2d_row_col(data: &[Complex64], ng1: usize, ng2: usize) -> FFTResult<Vec<Complex64>> {
    let mut buf = data.to_vec();

    // Transform along rows (axis 1, length ng2)
    for r in 0..ng1 {
        let row_start = r * ng2;
        let row: Vec<Complex64> = buf[row_start..row_start + ng2].to_vec();
        let row_fft = fft_internal(&row)?;
        buf[row_start..row_start + ng2].copy_from_slice(&row_fft);
    }

    // Transform along columns (axis 0, length ng1)
    for s in 0..ng2 {
        let col: Vec<Complex64> = (0..ng1).map(|r| buf[r * ng2 + s]).collect();
        let col_fft = fft_internal(&col)?;
        for (r, val) in col_fft.into_iter().enumerate() {
            buf[r * ng2 + s] = val;
        }
    }

    Ok(buf)
}

/// Row-column 2-D IFFT (in-place on flattened row-major buffer, normalised by ng1*ng2).
fn ifft2d_row_col(data: &[Complex64], ng1: usize, ng2: usize) -> FFTResult<Vec<Complex64>> {
    let mut buf = data.to_vec();
    let scale = 1.0 / (ng1 * ng2) as f64;

    // IFFT along rows first — note: we use the raw (unnormalised) FFT with conjugation
    // to implement IFFT: IFFT(x) = conj(FFT(conj(x))) / N
    for r in 0..ng1 {
        let row_start = r * ng2;
        let row: Vec<Complex64> = buf[row_start..row_start + ng2]
            .iter()
            .map(|c| c.conj())
            .collect();
        let row_fft = fft_internal(&row)?;
        for (s, val) in row_fft.into_iter().enumerate() {
            buf[row_start + s] = val.conj();
        }
    }

    // IFFT along columns
    for s in 0..ng2 {
        let col: Vec<Complex64> = (0..ng1).map(|r| buf[r * ng2 + s].conj()).collect();
        let col_fft = fft_internal(&col)?;
        for (r, val) in col_fft.into_iter().enumerate() {
            buf[r * ng2 + s] = val.conj() * scale;
        }
    }

    Ok(buf)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Generate `n` 2-D uniform points on a grid in `[-π, π)²`.
    fn uniform_grid(n1: usize, n2: usize) -> Vec<(f64, f64)> {
        let mut pts = Vec::with_capacity(n1 * n2);
        for i in 0..n1 {
            for j in 0..n2 {
                let x = -PI + 2.0 * PI * i as f64 / n1 as f64;
                let y = -PI + 2.0 * PI * j as f64 / n2 as f64;
                pts.push((x, y));
            }
        }
        pts
    }

    #[test]
    fn test_nufft2_type1_output_shape() {
        let pts = uniform_grid(8, 8);
        let c: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); pts.len()];
        let f_hat = nufft2_type1(&pts, &c, (8, 8), 1e-6).expect("type1");
        assert_eq!(f_hat.shape(), &[8, 8]);
    }

    #[test]
    fn test_nufft2_type2_output_length() {
        let n1 = 8usize;
        let n2 = 8usize;
        let f_hat = Array2::from_elem((n1, n2), Complex64::new(1.0, 0.0));
        let pts = uniform_grid(4, 4);
        let vals = nufft2_type2(&f_hat, &pts, 1e-6).expect("type2");
        assert_eq!(vals.len(), pts.len());
    }

    #[test]
    fn test_nufft2_type1_dc_impulse() {
        // If c[j] = 1 for all j and (x, y) span a uniform 2-D grid, then
        // only the DC mode (k1=0, k2=0) should be large.
        let n1 = 8usize;
        let n2 = 8usize;
        let pts = uniform_grid(n1, n2);
        let c: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); pts.len()];
        let f_hat = nufft2_type1(&pts, &c, (n1, n2), 1e-8).expect("type1");

        // DC is at index (n1/2, n2/2) in the centred convention
        let dc_mag = f_hat[(n1 / 2, n2 / 2)].norm();
        assert!(
            dc_mag > 0.5 * pts.len() as f64,
            "DC={:.3} expected ~{}",
            dc_mag,
            pts.len()
        );

        // Off-DC modes should be much smaller
        for r in 0..n1 {
            for s in 0..n2 {
                if r != n1 / 2 || s != n2 / 2 {
                    assert!(
                        f_hat[(r, s)].norm() < 0.25 * dc_mag,
                        "Off-DC mode ({},{}) too large: {:.3}",
                        r,
                        s,
                        f_hat[(r, s)].norm()
                    );
                }
            }
        }
    }

    #[test]
    fn test_nufft2_dimension_error() {
        let pts = vec![(-PI + 0.1, 0.0), (0.0, 0.0)];
        let c = vec![Complex64::new(1.0, 0.0)]; // length mismatch
        let res = nufft2_type1(&pts, &c, (4, 4), 1e-6);
        assert!(res.is_err());
    }

    #[test]
    fn test_nufft2_epsilon_error() {
        let pts = vec![(-PI + 0.1, 0.0)];
        let c = vec![Complex64::new(1.0, 0.0)];
        let res = nufft2_type1(&pts, &c, (4, 4), 0.0);
        assert!(res.is_err());
    }

    #[test]
    fn test_nufft2_range_error() {
        let pts = vec![(PI + 0.5, 0.0)]; // out of range
        let c = vec![Complex64::new(1.0, 0.0)];
        let res = nufft2_type1(&pts, &c, (4, 4), 1e-6);
        assert!(res.is_err());
    }

    #[test]
    fn test_nufft2_type1_single_tone() {
        // Source = exp(i·k₁·x + i·k₂·y): expect peak at mode (k1, k2).
        let n1 = 16usize;
        let n2 = 16usize;
        let k1_target: i64 = 2;
        let k2_target: i64 = 3;

        let pts = uniform_grid(n1, n2);
        let c: Vec<Complex64> = pts
            .iter()
            .map(|&(xj, yj)| {
                let phase = k1_target as f64 * xj + k2_target as f64 * yj;
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();

        let f_hat = nufft2_type1(&pts, &c, (n1, n2), 1e-8).expect("type1");

        let r_peak = (n1 / 2) as i64 + k1_target;
        let s_peak = (n2 / 2) as i64 + k2_target;
        let peak_mag = f_hat[(r_peak as usize, s_peak as usize)].norm();

        let max_other = f_hat
            .indexed_iter()
            .filter(|&((r, s), _)| r as i64 != r_peak || s as i64 != s_peak)
            .map(|(_, v)| v.norm())
            .fold(0.0f64, f64::max);

        assert!(
            peak_mag > 5.0 * max_other,
            "peak={:.3} max_other={:.3}",
            peak_mag,
            max_other
        );
    }

    #[test]
    fn test_nufft2_type2_constant_spectrum() {
        // If f_hat is all-ones, Type-2 should give a large DC response.
        let n1 = 8usize;
        let n2 = 8usize;
        let f_hat = Array2::from_elem((n1, n2), Complex64::new(1.0, 0.0));

        // Single target at origin
        let xy = vec![(0.0, 0.0)];
        let vals = nufft2_type2(&f_hat, &xy, 1e-6).expect("type2");
        assert_eq!(vals.len(), 1);
        // All ones → sum equals n1 * n2
        let expected = (n1 * n2) as f64;
        assert_relative_eq!(vals[0].re, expected, epsilon = 0.2 * expected);
    }

    #[test]
    fn test_nufft2_type2_empty_spectrum_error() {
        let f_hat: Array2<Complex64> = Array2::zeros((0, 4));
        let xy = vec![(0.0, 0.0)];
        let res = nufft2_type2(&f_hat, &xy, 1e-6);
        assert!(res.is_err());
    }

    #[test]
    fn test_fft2d_ifft2d_roundtrip() {
        // A 2-D FFT followed by IFFT should recover the original data.
        let ng1 = 4usize;
        let ng2 = 4usize;
        let data: Vec<Complex64> = (0..ng1 * ng2)
            .map(|i| Complex64::new(i as f64, -(i as f64) * 0.5))
            .collect();

        let fft_out = fft2d_row_col(&data, ng1, ng2).expect("fft");
        let recovered = ifft2d_row_col(&fft_out, ng1, ng2).expect("ifft");

        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig.re, rec.re, epsilon = 1e-10);
            assert_relative_eq!(orig.im, rec.im, epsilon = 1e-10);
        }
    }
}
