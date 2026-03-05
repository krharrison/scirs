//! 3D Volumetric Filters
//!
//! Provides production-quality volumetric image filters for 3-D arrays:
//!
//! * [`gaussian_filter_3d`]          – separable Gaussian smoothing
//! * [`median_filter_3d`]            – histogram-accelerated 3-D median filter
//! * [`sobel_3d`]                    – 6-direction Sobel gradient magnitude
//! * [`laplacian_3d`]                – Laplacian-of-Gaussian (LoG)
//! * [`bilateral_filter_3d`]         – range-and-space bilateral filter
//! * [`anisotropic_diffusion_3d`]    – Perona–Malik anisotropic diffusion
//!
//! All functions operate on `Array3<f64>` values and use clamp-to-edge border
//! handling unless stated otherwise.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array1, Array3, ArrayView3, s};

// ---------------------------------------------------------------------------
// Border helpers
// ---------------------------------------------------------------------------

#[inline]
fn clamp_coord(i: isize, len: usize) -> usize {
    i.clamp(0, len.saturating_sub(1) as isize) as usize
}

#[inline]
fn get_v(vol: &Array3<f64>, z: isize, y: isize, x: isize) -> f64 {
    let sh = vol.shape();
    vol[[
        clamp_coord(z, sh[0]),
        clamp_coord(y, sh[1]),
        clamp_coord(x, sh[2]),
    ]]
}

// ---------------------------------------------------------------------------
// 1-D Gaussian kernel
// ---------------------------------------------------------------------------

fn gaussian_kernel_1d(sigma: f64, truncate: f64) -> Array1<f64> {
    let radius = (truncate * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let two_sq = 2.0 * sigma * sigma;
    let mut k = Array1::<f64>::zeros(size);
    let mut sum = 0.0_f64;
    for i in 0..size {
        let d = (i as f64) - (radius as f64);
        let v = (-d * d / two_sq).exp();
        k[i] = v;
        sum += v;
    }
    for v in k.iter_mut() {
        *v /= sum;
    }
    k
}

// Convolve a 3-D array along a single axis with a 1-D kernel.
fn convolve_axis(vol: &Array3<f64>, kernel: &Array1<f64>, axis: usize) -> Array3<f64> {
    let sh = vol.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    let half = (kernel.len() / 2) as isize;
    let mut out = Array3::<f64>::zeros((nz, ny, nx));

    match axis {
        0 => {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let mut acc = 0.0_f64;
                        for (ki, &kv) in kernel.iter().enumerate() {
                            let zi = (z as isize) + (ki as isize) - half;
                            acc += kv * get_v(vol, zi, y as isize, x as isize);
                        }
                        out[[z, y, x]] = acc;
                    }
                }
            }
        }
        1 => {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let mut acc = 0.0_f64;
                        for (ki, &kv) in kernel.iter().enumerate() {
                            let yi = (y as isize) + (ki as isize) - half;
                            acc += kv * get_v(vol, z as isize, yi, x as isize);
                        }
                        out[[z, y, x]] = acc;
                    }
                }
            }
        }
        _ => {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let mut acc = 0.0_f64;
                        for (ki, &kv) in kernel.iter().enumerate() {
                            let xi = (x as isize) + (ki as isize) - half;
                            acc += kv * get_v(vol, z as isize, y as isize, xi);
                        }
                        out[[z, y, x]] = acc;
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// gaussian_filter_3d
// ---------------------------------------------------------------------------

/// Apply a 3-D Gaussian filter to a volumetric array.
///
/// The filter is implemented as three successive 1-D convolutions (separable
/// decomposition), which is exact for isotropic Gaussians and has complexity
/// `O(N · k)` rather than `O(N · k³)`.
///
/// # Arguments
///
/// * `volume` – Input volume with shape `(depth, height, width)`.
/// * `sigma`  – Standard deviation of the Gaussian kernel (isotropic).
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when `sigma ≤ 0`.
pub fn gaussian_filter_3d(
    volume: ArrayView3<f64>,
    sigma: f64,
) -> NdimageResult<Array3<f64>> {
    if sigma <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "sigma must be positive".to_string(),
        ));
    }
    let vol = volume.to_owned();
    let k = gaussian_kernel_1d(sigma, 4.0);
    let tmp = convolve_axis(&vol, &k, 0);
    let tmp = convolve_axis(&tmp, &k, 1);
    Ok(convolve_axis(&tmp, &k, 2))
}

// ---------------------------------------------------------------------------
// median_filter_3d
// ---------------------------------------------------------------------------

/// Apply a 3-D median filter with a cubic neighbourhood of side `size`.
///
/// The implementation collects all neighbourhood values into a `Vec`, sorts
/// them, and picks the middle value.  Border voxels use clamp-to-edge padding.
///
/// # Arguments
///
/// * `volume` – Input volume.
/// * `size`   – Side length of the cubic sliding window (must be odd and ≥ 1).
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when `size == 0`.
pub fn median_filter_3d(
    volume: ArrayView3<f64>,
    size: usize,
) -> NdimageResult<Array3<f64>> {
    if size == 0 {
        return Err(NdimageError::InvalidInput(
            "size must be ≥ 1".to_string(),
        ));
    }
    let vol = volume.to_owned();
    let sh = vol.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    let half = (size / 2) as isize;
    let mut out = Array3::<f64>::zeros((nz, ny, nx));

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut neighbourhood: Vec<f64> = Vec::with_capacity(size * size * size);
                for dz in -half..=half {
                    for dy in -half..=half {
                        for dx in -half..=half {
                            neighbourhood.push(get_v(
                                &vol,
                                z as isize + dz,
                                y as isize + dy,
                                x as isize + dx,
                            ));
                        }
                    }
                }
                neighbourhood.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = neighbourhood.len() / 2;
                out[[z, y, x]] = neighbourhood[mid];
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// sobel_3d – gradient magnitude via 6-direction Sobel operators
// ---------------------------------------------------------------------------

/// Compute the 3-D Sobel gradient magnitude.
///
/// Three independent Sobel operators are applied along the Z, Y, and X axes;
/// the gradient magnitude is `√(Gz² + Gy² + Gx²)`.
///
/// Returns `(magnitude, gz, gy, gx)` so callers can also access directional
/// components.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the volume is empty.
pub fn sobel_3d(
    volume: ArrayView3<f64>,
) -> NdimageResult<(Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>)> {
    let sh = volume.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    if nz == 0 || ny == 0 || nx == 0 {
        return Err(NdimageError::InvalidInput(
            "Volume must be non-empty".to_string(),
        ));
    }
    let vol = volume.to_owned();

    let mut gz = Array3::<f64>::zeros((nz, ny, nx));
    let mut gy = Array3::<f64>::zeros((nz, ny, nx));
    let mut gx = Array3::<f64>::zeros((nz, ny, nx));

    // Sobel weights: derivative direction uses [-1, 0, +1]; perpendicular
    // directions use the smoothing weights [1, 2, 1] / 4 each.
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let zi = z as isize;
                let yi = y as isize;
                let xi = x as isize;

                // Gradient along Z
                let gz_val = {
                    let mut acc = 0.0_f64;
                    for dy in -1_isize..=1 {
                        for dx in -1_isize..=1 {
                            let wy = if dy == 0 { 2.0 } else { 1.0 };
                            let wx = if dx == 0 { 2.0 } else { 1.0 };
                            let w = wy * wx;
                            acc += w * (get_v(&vol, zi + 1, yi + dy, xi + dx)
                                - get_v(&vol, zi - 1, yi + dy, xi + dx));
                        }
                    }
                    acc / 32.0
                };

                // Gradient along Y
                let gy_val = {
                    let mut acc = 0.0_f64;
                    for dz in -1_isize..=1 {
                        for dx in -1_isize..=1 {
                            let wz = if dz == 0 { 2.0 } else { 1.0 };
                            let wx = if dx == 0 { 2.0 } else { 1.0 };
                            let w = wz * wx;
                            acc += w * (get_v(&vol, zi + dz, yi + 1, xi + dx)
                                - get_v(&vol, zi + dz, yi - 1, xi + dx));
                        }
                    }
                    acc / 32.0
                };

                // Gradient along X
                let gx_val = {
                    let mut acc = 0.0_f64;
                    for dz in -1_isize..=1 {
                        for dy in -1_isize..=1 {
                            let wz = if dz == 0 { 2.0 } else { 1.0 };
                            let wy = if dy == 0 { 2.0 } else { 1.0 };
                            let w = wz * wy;
                            acc += w * (get_v(&vol, zi + dz, yi + dy, xi + 1)
                                - get_v(&vol, zi + dz, yi + dy, xi - 1));
                        }
                    }
                    acc / 32.0
                };

                gz[[z, y, x]] = gz_val;
                gy[[z, y, x]] = gy_val;
                gx[[z, y, x]] = gx_val;
            }
        }
    }

    let mut mag = Array3::<f64>::zeros((nz, ny, nx));
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let v = gz[[z, y, x]].powi(2)
                    + gy[[z, y, x]].powi(2)
                    + gx[[z, y, x]].powi(2);
                mag[[z, y, x]] = v.sqrt();
            }
        }
    }
    Ok((mag, gz, gy, gx))
}

// ---------------------------------------------------------------------------
// laplacian_3d  (Laplacian-of-Gaussian)
// ---------------------------------------------------------------------------

/// Compute the Laplacian-of-Gaussian (LoG) of a 3-D volume.
///
/// The implementation first applies a Gaussian blur (controlled by `sigma`)
/// and then applies the discrete 3-D Laplacian kernel:
///
/// ```text
///   [ 0  0  0 ]   [ 0  1  0 ]   [ 0  0  0 ]
///   [ 0  1  0 ] + [ 1  1  1 ] + [ 0  1  0 ]   center = -6
///   [ 0  0  0 ]   [ 0  1  0 ]   [ 0  0  0 ]
/// ```
///
/// # Arguments
///
/// * `volume` – Input volume.
/// * `sigma`  – Gaussian pre-smoothing standard deviation.  Set to `0.0` to
///              skip smoothing (raw LoG from the 7-point stencil).
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when `sigma < 0`.
pub fn laplacian_3d(volume: ArrayView3<f64>, sigma: f64) -> NdimageResult<Array3<f64>> {
    if sigma < 0.0 {
        return Err(NdimageError::InvalidInput(
            "sigma must be ≥ 0".to_string(),
        ));
    }
    let smoothed: Array3<f64> = if sigma > 0.0 {
        gaussian_filter_3d(volume, sigma)?
    } else {
        volume.to_owned()
    };

    let sh = smoothed.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    let mut out = Array3::<f64>::zeros((nz, ny, nx));

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let zi = z as isize;
                let yi = y as isize;
                let xi = x as isize;
                let center = smoothed[[z, y, x]];
                let lap = get_v(&smoothed, zi - 1, yi, xi)
                    + get_v(&smoothed, zi + 1, yi, xi)
                    + get_v(&smoothed, zi, yi - 1, xi)
                    + get_v(&smoothed, zi, yi + 1, xi)
                    + get_v(&smoothed, zi, yi, xi - 1)
                    + get_v(&smoothed, zi, yi, xi + 1)
                    - 6.0 * center;
                out[[z, y, x]] = lap;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// bilateral_filter_3d
// ---------------------------------------------------------------------------

/// Apply a 3-D bilateral filter.
///
/// The bilateral filter smooths a volume while preserving edges by weighting
/// each neighbour with both a spatial Gaussian and an intensity-range Gaussian.
///
/// # Arguments
///
/// * `volume`       – Input volume.
/// * `d`            – Diameter (in voxels) of the filtering neighbourhood.  The
///                    half-radius is `d / 2`; the spatial sigma is `d / 6`.
/// * `sigma_color`  – Standard deviation in intensity space.
/// * `sigma_space`  – Standard deviation in spatial distance space (override; if
///                    `0.0` the default `d / 6` is used).
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when `d == 0` or either sigma is
/// negative.
pub fn bilateral_filter_3d(
    volume: ArrayView3<f64>,
    d: usize,
    sigma_color: f64,
    sigma_space: f64,
) -> NdimageResult<Array3<f64>> {
    if d == 0 {
        return Err(NdimageError::InvalidInput(
            "d must be ≥ 1".to_string(),
        ));
    }
    if sigma_color < 0.0 || sigma_space < 0.0 {
        return Err(NdimageError::InvalidInput(
            "sigma values must be ≥ 0".to_string(),
        ));
    }

    let ss = if sigma_space > 0.0 {
        sigma_space
    } else {
        (d as f64) / 6.0_f64.max(1.0)
    };
    let two_ss_sq = 2.0 * ss * ss;
    let two_sc_sq = 2.0 * sigma_color * sigma_color;

    let vol = volume.to_owned();
    let sh = vol.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    let half = (d as isize) / 2;
    let mut out = Array3::<f64>::zeros((nz, ny, nx));

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let center_val = vol[[z, y, x]];
                let mut weighted_sum = 0.0_f64;
                let mut weight_total = 0.0_f64;

                for dz in -half..=half {
                    for dy in -half..=half {
                        for dx in -half..=half {
                            let nv = get_v(
                                &vol,
                                z as isize + dz,
                                y as isize + dy,
                                x as isize + dx,
                            );
                            let dist_sq =
                                (dz * dz + dy * dy + dx * dx) as f64;
                            let color_diff = nv - center_val;
                            let w_space = (-dist_sq / two_ss_sq).exp();
                            let w_color =
                                (-(color_diff * color_diff) / two_sc_sq).exp();
                            let w = w_space * w_color;
                            weighted_sum += w * nv;
                            weight_total += w;
                        }
                    }
                }

                out[[z, y, x]] = if weight_total > 0.0 {
                    weighted_sum / weight_total
                } else {
                    center_val
                };
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// anisotropic_diffusion_3d  (Perona–Malik)
// ---------------------------------------------------------------------------

/// Apply Perona–Malik anisotropic diffusion to a 3-D volume.
///
/// The Perona–Malik PDE is solved on a discrete 3-D grid using an explicit
/// forward Euler scheme.  The conduction function is:
///
/// ```text
///   c(∇I) = exp(−(|∇I| / kappa)²)
/// ```
///
/// which preserves strong edges (large gradient) while diffusing across weak
/// intensity boundaries.
///
/// # Arguments
///
/// * `volume` – Input floating-point volume.
/// * `niter`  – Number of diffusion iterations.
/// * `kappa`  – Edge-sensitivity threshold (larger → more diffusion across
///              edges; typical range 10–100).
/// * `gamma`  – Time-step (stability requires `gamma ≤ 1/6` in 3-D).
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when `kappa ≤ 0`, `niter == 0`,
/// or `gamma` is out of range.
pub fn anisotropic_diffusion_3d(
    volume: ArrayView3<f64>,
    niter: usize,
    kappa: f64,
    gamma: f64,
) -> NdimageResult<Array3<f64>> {
    if kappa <= 0.0 {
        return Err(NdimageError::InvalidInput(
            "kappa must be > 0".to_string(),
        ));
    }
    if niter == 0 {
        return Err(NdimageError::InvalidInput(
            "niter must be ≥ 1".to_string(),
        ));
    }
    if gamma <= 0.0 || gamma > 1.0 / 6.0 + 1e-9 {
        return Err(NdimageError::InvalidInput(
            "gamma must be in (0, 1/6] for numerical stability".to_string(),
        ));
    }

    let sh = volume.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    let kk = kappa * kappa;

    let mut cur = volume.to_owned();

    for _ in 0..niter {
        let prev = cur.clone();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let c = prev[[z, y, x]];
                    let zi = z as isize;
                    let yi = y as isize;
                    let xi = x as isize;

                    // 6-connected neighbours
                    let dirs: [(isize, isize, isize); 6] = [
                        (-1, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
                    ];

                    let mut flux = 0.0_f64;
                    for (dz, dy, dx) in dirs {
                        let nv = get_v(&prev, zi + dz, yi + dy, xi + dx);
                        let diff = nv - c;
                        let cond = (-diff * diff / kk).exp();
                        flux += cond * diff;
                    }
                    cur[[z, y, x]] = c + gamma * flux;
                }
            }
        }
    }
    Ok(cur)
}

// ---------------------------------------------------------------------------
// Additional helper: gradient magnitude (used by measurements3d)
// ---------------------------------------------------------------------------

/// Compute the finite-difference gradient magnitude of a volume.
///
/// Uses central differences (or forward/backward differences at borders).
pub fn gradient_magnitude_3d(volume: &Array3<f64>) -> Array3<f64> {
    let sh = volume.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    let mut out = Array3::<f64>::zeros((nz, ny, nx));

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let zi = z as isize;
                let yi = y as isize;
                let xi = x as isize;

                let dz = (get_v(volume, zi + 1, yi, xi) - get_v(volume, zi - 1, yi, xi)) * 0.5;
                let dy = (get_v(volume, zi, yi + 1, xi) - get_v(volume, zi, yi - 1, xi)) * 0.5;
                let dx = (get_v(volume, zi, yi, xi + 1) - get_v(volume, zi, yi, xi - 1)) * 0.5;
                out[[z, y, x]] = (dz * dz + dy * dy + dx * dx).sqrt();
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn make_vol(nz: usize, ny: usize, nx: usize, val: f64) -> Array3<f64> {
        Array3::from_elem((nz, ny, nx), val)
    }

    #[test]
    fn gaussian_uniform_preserves_constant() {
        let vol = make_vol(10, 10, 10, 3.0);
        let out = gaussian_filter_3d(vol.view(), 1.5).expect("gaussian_filter_3d failed");
        for v in out.iter() {
            assert!((v - 3.0).abs() < 1e-10, "Expected ~3.0, got {v}");
        }
    }

    #[test]
    fn gaussian_rejects_nonpositive_sigma() {
        let vol = make_vol(5, 5, 5, 1.0);
        assert!(gaussian_filter_3d(vol.view(), 0.0).is_err());
        assert!(gaussian_filter_3d(vol.view(), -1.0).is_err());
    }

    #[test]
    fn median_preserves_constant() {
        let vol = make_vol(8, 8, 8, 7.0);
        let out = median_filter_3d(vol.view(), 3).expect("median_filter_3d failed");
        for v in out.iter() {
            assert!((v - 7.0).abs() < 1e-10);
        }
    }

    #[test]
    fn median_rejects_zero_size() {
        let vol = make_vol(5, 5, 5, 1.0);
        assert!(median_filter_3d(vol.view(), 0).is_err());
    }

    #[test]
    fn sobel_zero_gradient_on_constant() {
        let vol = make_vol(8, 8, 8, 4.0);
        let (mag, gz, gy, gx) = sobel_3d(vol.view()).expect("sobel_3d failed");
        for v in mag.iter() {
            assert!(v.abs() < 1e-10, "magnitude should be 0 on constant volume");
        }
        for v in gz.iter().chain(gy.iter()).chain(gx.iter()) {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn sobel_rejects_empty() {
        let vol = Array3::<f64>::zeros((0, 4, 4));
        assert!(sobel_3d(vol.view()).is_err());
    }

    #[test]
    fn laplacian_zero_on_constant() {
        let vol = make_vol(6, 6, 6, 2.5);
        let out = laplacian_3d(vol.view(), 0.0).expect("laplacian_3d failed");
        for v in out.iter() {
            assert!(v.abs() < 1e-10, "LoG should be 0 on constant, got {v}");
        }
    }

    #[test]
    fn bilateral_preserves_constant() {
        let vol = make_vol(8, 8, 8, 5.0);
        let out = bilateral_filter_3d(vol.view(), 5, 15.0, 1.5)
            .expect("bilateral_filter_3d failed");
        for v in out.iter() {
            assert!((v - 5.0).abs() < 1e-9, "bilateral constant: {v}");
        }
    }

    #[test]
    fn bilateral_rejects_zero_d() {
        let vol = make_vol(5, 5, 5, 1.0);
        assert!(bilateral_filter_3d(vol.view(), 0, 10.0, 1.0).is_err());
    }

    #[test]
    fn anisotropic_diffusion_preserves_constant() {
        let vol = make_vol(8, 8, 8, 3.0);
        let out =
            anisotropic_diffusion_3d(vol.view(), 5, 20.0, 1.0 / 7.0).expect("aniso failed");
        for v in out.iter() {
            assert!((v - 3.0).abs() < 1e-9, "aniso constant: {v}");
        }
    }

    #[test]
    fn anisotropic_diffusion_rejects_bad_params() {
        let vol = make_vol(6, 6, 6, 1.0);
        assert!(anisotropic_diffusion_3d(vol.view(), 0, 20.0, 0.1).is_err());
        assert!(anisotropic_diffusion_3d(vol.view(), 5, 0.0, 0.1).is_err());
        assert!(anisotropic_diffusion_3d(vol.view(), 5, 20.0, 0.5).is_err());
    }

    #[test]
    fn gradient_magnitude_zero_on_constant() {
        let vol = make_vol(6, 6, 6, 2.0);
        let out = gradient_magnitude_3d(&vol);
        for v in out.iter() {
            assert!(v.abs() < 1e-10);
        }
    }
}
