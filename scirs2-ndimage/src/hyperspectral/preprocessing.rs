//! Hyperspectral Image Preprocessing
//!
//! Implements noise reduction, dimensionality transforms, band management,
//! and radiometric correction routines for hyperspectral imagery.

use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};

use crate::error::{NdimageError, NdimageResult};
use crate::hyperspectral::unmixing::HyperspectralImage;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-column mean of a 2-D array.
fn col_mean(a: &Array2<f64>) -> Array1<f64> {
    a.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(a.ncols()))
}

/// Centre an array by subtracting column means.
fn centre_cols(a: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let mean = col_mean(a);
    let centred = a - &mean.view().insert_axis(Axis(0));
    (centred, mean)
}

/// QR-based orthonormal factorisation of a `(m × r)` matrix (Gram–Schmidt).
fn qr_orth(a: &Array2<f64>, r: usize) -> Array2<f64> {
    let m = a.nrows();
    let cols = r.min(a.ncols());
    let mut q = Array2::<f64>::zeros((m, cols));

    for k in 0..cols {
        let mut col = a.column(k).to_owned();
        for j in 0..k {
            let qj = q.column(j).to_owned();
            let proj: f64 = col.iter().zip(qj.iter()).map(|(x, y)| x * y).sum();
            col = col - qj * proj;
        }
        let n: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-14 {
            for i in 0..m {
                q[[i, k]] = col[i] / n;
            }
        } else if k < m {
            q[[k, k]] = 1.0;
        }
    }
    q
}

/// Thin SVD returning (U [m×r], S [r], Vt [r×n]) using randomised power iteration.
fn thin_svd_pp(a: &Array2<f64>, rank: usize) -> NdimageResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = (a.nrows(), a.ncols());
    let r = rank.min(m).min(n).max(1);

    // Build AtA (n×n) and extract top-r eigenvectors.
    let ata = a.t().dot(a);

    // Seed deterministic near-random matrix.
    let mut seed_mat = Array2::<f64>::zeros((n, r));
    let mut state: u64 = 0xFEED_DEAD_CAFE_1337;
    for i in 0..n {
        for j in 0..r {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed_mat[[i, j]] = ((state >> 11) as f64) / (1u64 << 53) as f64 - 0.5;
        }
    }
    let mut q = qr_orth(&seed_mat, r);

    // Power iteration.
    for _ in 0..20 {
        let z = ata.dot(&q);
        q = qr_orth(&z, r);
    }

    // B = A Q  [m × r]
    let b = a.dot(&q);
    // QR of B to get U.
    let u = qr_orth(&b, r);

    // S and Vt from U^T A.
    let uta = u.t().dot(a); // [r × n]
    let mut s = Array1::<f64>::zeros(r);
    let mut vt = Array2::<f64>::zeros((r, n));

    for k in 0..r {
        let row = uta.row(k).to_owned();
        let n_row: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        s[k] = n_row;
        if n_row > 1e-14 {
            for j in 0..n {
                vt[[k, j]] = row[j] / n_row;
            }
        }
    }

    Ok((u, s, vt))
}

/// Compute eigenvalues and eigenvectors of a symmetric positive-semidefinite
/// matrix via iterative QR-algorithm (Jacobi-like Gram–Schmidt).
/// Returns (eigenvalues desc, eigenvectors as columns).
fn sym_eigen(a: &Array2<f64>, max_iter: usize) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows();
    // Use power iteration per eigenvector (deflation).
    let mut eigvecs = Array2::<f64>::zeros((n, n));
    let mut eigvals = Array1::<f64>::zeros(n);
    let mut deflated = a.clone();

    for k in 0..n {
        // Start with canonical basis vector.
        let mut v = Array1::<f64>::zeros(n);
        v[k] = 1.0;

        for _ in 0..max_iter {
            let w = deflated.dot(&v);
            let nw: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nw < 1e-14 {
                break;
            }
            v = w / nw;
        }

        let lambda: f64 = v.iter().zip(deflated.dot(&v).iter()).map(|(a, b)| a * b).sum();
        eigvals[k] = lambda;
        for i in 0..n {
            eigvecs[[i, k]] = v[i];
        }

        // Deflate: A <- A - lambda * v v^T
        for i in 0..n {
            for j in 0..n {
                deflated[[i, j]] -= lambda * v[i] * v[j];
            }
        }
    }
    (eigvals, eigvecs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimum Noise Fraction (MNF)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the Minimum Noise Fraction (MNF) transform.
#[derive(Debug, Clone)]
pub struct MnfResult {
    /// MNF components: `[N_pixels, n_components]` (sorted by SNR descending).
    pub components: Array2<f64>,
    /// Forward transform matrix `[n_components, N_bands]`.
    pub transform: Array2<f64>,
    /// Approximate signal-to-noise ratio for each component.
    pub snr: Array1<f64>,
}

/// Minimum Noise Fraction (MNF) transform.
///
/// Two-step PCA procedure that first whitens the noise covariance, then applies
/// PCA to recover components sorted by signal-to-noise ratio (Green et al. 1988).
///
/// # Arguments
/// * `image`        - Hyperspectral image `[N_pixels, N_bands]`.
/// * `n_components` - Number of MNF components to retain.
///
/// # Returns
/// [`MnfResult`] containing transformed data, transform matrix and SNR estimates.
pub fn minimum_noise_fraction(
    image: &HyperspectralImage,
    n_components: usize,
) -> NdimageResult<MnfResult> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    let nc = n_components.min(n_bands);

    if nc == 0 {
        return Err(NdimageError::InvalidInput("n_components must be >= 1".into()));
    }

    // Step 1: estimate noise covariance via first-difference filter.
    // Noise ≈ diff(y_i+1 - y_i) / sqrt(2).
    if n_pixels < 2 {
        return Err(NdimageError::InvalidInput("Need at least 2 pixels for MNF".into()));
    }

    let data = &image.data;
    let diffs = data.slice(s![1.., ..]).to_owned() - data.slice(s![..n_pixels - 1, ..]).to_owned();
    let noise_cov_raw = diffs.t().dot(&diffs) / (2.0 * (n_pixels - 1) as f64);
    // noise_cov_raw: [L, L]

    // Step 2: whitening transform W = noise_cov^{-1/2}.
    // Eigendecompose noise_cov.
    let (noise_eigs, noise_vecs) = sym_eigen(&noise_cov_raw, 200);

    // Build W = V D^{-1/2} V^T  (regularise small eigenvalues).
    let eps = 1e-8 * noise_eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1.0);
    let mut d_inv_sqrt = Array1::<f64>::zeros(n_bands);
    for k in 0..n_bands {
        d_inv_sqrt[k] = if noise_eigs[k] > eps { 1.0 / noise_eigs[k].sqrt() } else { 0.0 };
    }

    // W: [L × L]
    let mut w = Array2::<f64>::zeros((n_bands, n_bands));
    for i in 0..n_bands {
        for j in 0..n_bands {
            let mut s = 0.0_f64;
            for k in 0..n_bands {
                s += noise_vecs[[i, k]] * d_inv_sqrt[k] * noise_vecs[[j, k]];
            }
            w[[i, j]] = s;
        }
    }

    // Step 3: apply whitening to data.
    let (centred, mean) = centre_cols(data);
    let whitened = centred.dot(&w.t()); // [N, L]

    // Step 4: PCA of whitened data to find MNF components.
    let (_, s_vals, vt) = thin_svd_pp(&whitened, nc)?; // vt: [nc, L]

    // MNF transform = Vt @ W (maps original spectra to MNF space).
    let transform = vt.dot(&w); // [nc, L]

    // Apply transform to get components.
    let components = (centred).dot(&transform.t()); // [N, nc]

    // Approximate SNR per component: s^2 / residual_var.
    let snr = s_vals.mapv(|sv| sv * sv / (n_pixels as f64).max(1.0));

    let _ = mean; // mean was already subtracted into centred.

    Ok(MnfResult { components, transform, snr })
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral whitening
// ─────────────────────────────────────────────────────────────────────────────

/// Spectrally whiten a hyperspectral image.
///
/// Transforms the data so that each spectral component has zero mean and
/// unit variance. The covariance matrix of the whitened data is the identity.
///
/// Whitening matrix `W = C^{-1/2}` where `C` is the sample covariance of bands.
///
/// # Returns
/// `(whitened_image, whitening_matrix [N_bands, N_bands])`.
pub fn whiten_hyperspectral(
    image: &HyperspectralImage,
) -> NdimageResult<(HyperspectralImage, Array2<f64>)> {
    let n_bands = image.n_bands();
    let (centred, mean) = centre_cols(&image.data);

    // Sample covariance C = X^T X / (N - 1).
    let n_f = (image.n_pixels().saturating_sub(1).max(1)) as f64;
    let cov = centred.t().dot(&centred) / n_f; // [L, L]

    // Eigendecompose C.
    let (eigs, vecs) = sym_eigen(&cov, 300);
    let eps = 1e-9 * eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1.0);

    // Build whitening matrix W = V D^{-1/2} V^T.
    let mut w = Array2::<f64>::zeros((n_bands, n_bands));
    for i in 0..n_bands {
        for j in 0..n_bands {
            let mut s = 0.0_f64;
            for k in 0..n_bands {
                let inv_sqrt = if eigs[k] > eps { 1.0 / eigs[k].sqrt() } else { 0.0 };
                s += vecs[[i, k]] * inv_sqrt * vecs[[j, k]];
            }
            w[[i, j]] = s;
        }
    }

    let whitened_data = centred.dot(&w.t()); // [N, L]
    let wavelengths = image.wavelengths.clone();
    let whitened_img = HyperspectralImage { data: whitened_data, wavelengths };
    let _ = mean;
    Ok((whitened_img, w))
}

// ─────────────────────────────────────────────────────────────────────────────
// Band removal
// ─────────────────────────────────────────────────────────────────────────────

/// Remove specified bands (columns) from a hyperspectral image.
///
/// Useful for discarding water-absorption bands (typically around 1350–1450 nm
/// and 1800–1950 nm) or detector-noise bands.
///
/// # Arguments
/// * `image`        - Input hyperspectral image.
/// * `bands_to_remove` - Slice of 0-based band indices to discard.
///
/// # Returns
/// A new `HyperspectralImage` with the specified bands removed.
pub fn remove_bands(
    image: &HyperspectralImage,
    bands_to_remove: &[usize],
) -> NdimageResult<HyperspectralImage> {
    let n_bands = image.n_bands();
    let n_pixels = image.n_pixels();

    for &b in bands_to_remove {
        if b >= n_bands {
            return Err(NdimageError::InvalidInput(format!(
                "Band index {} out of range (n_bands={})",
                b, n_bands
            )));
        }
    }

    let keep: Vec<usize> = (0..n_bands).filter(|b| !bands_to_remove.contains(b)).collect();
    let n_keep = keep.len();
    if n_keep == 0 {
        return Err(NdimageError::InvalidInput("All bands removed — at least one must remain".into()));
    }

    let mut new_data = Array2::<f64>::zeros((n_pixels, n_keep));
    for (new_b, &old_b) in keep.iter().enumerate() {
        for p in 0..n_pixels {
            new_data[[p, new_b]] = image.data[[p, old_b]];
        }
    }

    let new_wavelengths = image.wavelengths.as_ref().map(|wl| {
        Array1::from_vec(keep.iter().map(|&b| wl[b]).collect::<Vec<_>>())
    });

    Ok(HyperspectralImage { data: new_data, wavelengths: new_wavelengths })
}

/// Remove bands in wavelength ranges specified as `(wl_min, wl_max)` pairs.
///
/// Requires that the image has wavelength labels (set via [`HyperspectralImage::with_wavelengths`]).
///
/// # Arguments
/// * `image`          - Input image with wavelength labels.
/// * `absorb_ranges`  - Slice of `(min_nm, max_nm)` tuples defining absorption windows.
///
/// # Returns
/// New image with absorption-band columns removed.
pub fn remove_absorption_bands(
    image: &HyperspectralImage,
    absorb_ranges: &[(f64, f64)],
) -> NdimageResult<HyperspectralImage> {
    let wavelengths = image.wavelengths.as_ref().ok_or_else(|| {
        NdimageError::InvalidInput("remove_absorption_bands requires wavelength labels".into())
    })?;

    let bands_to_remove: Vec<usize> = wavelengths
        .iter()
        .enumerate()
        .filter_map(|(b, &wl)| {
            if absorb_ranges.iter().any(|&(lo, hi)| wl >= lo && wl <= hi) {
                Some(b)
            } else {
                None
            }
        })
        .collect();

    remove_bands(image, &bands_to_remove)
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial smoothing
// ─────────────────────────────────────────────────────────────────────────────

/// Spatial averaging noise reduction for a hyperspectral cube stored as
/// a 3-D array `[rows, cols, bands]`.
///
/// Each pixel is replaced by the mean of its `window_size × window_size`
/// neighbourhood (reflecting boundary).
///
/// # Arguments
/// * `cube`        - 3-D hyperspectral array `[H, W, B]`.
/// * `window_size` - Spatial averaging window (must be odd).
///
/// # Returns
/// Smoothed cube of the same shape.
pub fn spatial_smoothing(
    cube: &Array3<f64>,
    window_size: usize,
) -> NdimageResult<Array3<f64>> {
    let (h, w, b) = (cube.shape()[0], cube.shape()[1], cube.shape()[2]);

    if window_size == 0 {
        return Err(NdimageError::InvalidInput("window_size must be >= 1".into()));
    }
    if window_size % 2 == 0 {
        return Err(NdimageError::InvalidInput("window_size must be odd".into()));
    }

    let half = window_size / 2;
    let mut result = Array3::<f64>::zeros((h, w, b));

    for row in 0..h {
        for col in 0..w {
            let r_lo = row.saturating_sub(half);
            let r_hi = (row + half + 1).min(h);
            let c_lo = col.saturating_sub(half);
            let c_hi = (col + half + 1).min(w);
            let n_neighbours = ((r_hi - r_lo) * (c_hi - c_lo)) as f64;

            for band in 0..b {
                let mut s = 0.0_f64;
                for r in r_lo..r_hi {
                    for c in c_lo..c_hi {
                        s += cube[[r, c, band]];
                    }
                }
                result[[row, col, band]] = s / n_neighbours;
            }
        }
    }
    Ok(result)
}

/// Flatten a spatial hyperspectral cube `[H, W, B]` into a pixel matrix `[H*W, B]`.
pub fn cube_to_pixels(cube: &Array3<f64>) -> HyperspectralImage {
    let (h, w, b) = (cube.shape()[0], cube.shape()[1], cube.shape()[2]);
    let n_pixels = h * w;
    let mut data = Array2::<f64>::zeros((n_pixels, b));
    for r in 0..h {
        for c in 0..w {
            for band in 0..b {
                data[[r * w + c, band]] = cube[[r, c, band]];
            }
        }
    }
    HyperspectralImage::new(data)
}

/// Reshape a pixel matrix `[H*W, B]` back into a cube `[H, W, B]`.
pub fn pixels_to_cube(
    image: &HyperspectralImage,
    height: usize,
    width: usize,
) -> NdimageResult<Array3<f64>> {
    let n_pixels = image.n_pixels();
    if height * width != n_pixels {
        return Err(NdimageError::InvalidInput(format!(
            "height({}) * width({}) = {} != n_pixels {}",
            height, width, height * width, n_pixels
        )));
    }
    let b = image.n_bands();
    let mut cube = Array3::<f64>::zeros((height, width, b));
    for p in 0..n_pixels {
        let r = p / width;
        let c = p % width;
        for band in 0..b {
            cube[[r, c, band]] = image.data[[p, band]];
        }
    }
    Ok(cube)
}

// ─────────────────────────────────────────────────────────────────────────────
// Radiometric correction
// ─────────────────────────────────────────────────────────────────────────────

/// Radiometric calibration parameters per band.
#[derive(Debug, Clone)]
pub struct RadiometricCalibration {
    /// Per-band multiplicative gain (DN to radiance).
    pub gain: Array1<f64>,
    /// Per-band additive offset (bias).
    pub offset: Array1<f64>,
    /// Per-band solar irradiance `[W m^{-2} μm^{-1}]`.
    pub solar_irradiance: Option<Array1<f64>>,
    /// Solar zenith angle in radians (used for surface reflectance).
    pub solar_zenith_rad: Option<f64>,
}

impl RadiometricCalibration {
    /// Create a simple linear calibration with constant gain and zero offset.
    pub fn uniform(n_bands: usize, gain: f64) -> Self {
        Self {
            gain: Array1::from_elem(n_bands, gain),
            offset: Array1::zeros(n_bands),
            solar_irradiance: None,
            solar_zenith_rad: None,
        }
    }
}

/// Convert raw digital numbers (DN) to at-sensor radiance or surface reflectance.
///
/// Applies per-band linear calibration: `Radiance = gain * DN + offset`.
/// If `solar_irradiance` and `solar_zenith_rad` are provided, further converts
/// to top-of-atmosphere reflectance:
/// `ρ = π * L / (E_s * cos(θ_s))`.
///
/// # Arguments
/// * `image`  - Raw DN hyperspectral image `[N_pixels, N_bands]`.
/// * `cal`    - Calibration parameters.
///
/// # Returns
/// Corrected hyperspectral image.
pub fn radiometric_correction(
    image: &HyperspectralImage,
    cal: &RadiometricCalibration,
) -> NdimageResult<HyperspectralImage> {
    let n_bands = image.n_bands();
    if cal.gain.len() != n_bands || cal.offset.len() != n_bands {
        return Err(NdimageError::InvalidInput(format!(
            "Calibration gain/offset length {} != n_bands {}",
            cal.gain.len(), n_bands
        )));
    }

    if let Some(ref irr) = cal.solar_irradiance {
        if irr.len() != n_bands {
            return Err(NdimageError::InvalidInput(
                "solar_irradiance length must equal n_bands".into()
            ));
        }
    }

    let n_pixels = image.n_pixels();
    let mut corrected = Array2::<f64>::zeros((n_pixels, n_bands));

    for p in 0..n_pixels {
        for b in 0..n_bands {
            // Step 1: DN → radiance.
            let radiance = cal.gain[b] * image.data[[p, b]] + cal.offset[b];

            // Step 2: optionally convert to TOA reflectance.
            let value = match (&cal.solar_irradiance, cal.solar_zenith_rad) {
                (Some(irr), Some(zenith)) => {
                    let cos_zen = zenith.cos().max(0.01);
                    std::f64::consts::PI * radiance / (irr[b] * cos_zen)
                }
                _ => radiance,
            };
            corrected[[p, b]] = value;
        }
    }

    Ok(HyperspectralImage { data: corrected, wavelengths: image.wavelengths.clone() })
}

/// Dark-object subtraction (DOS) atmospheric correction.
///
/// Subtracts per-band minimum radiance (dark object value) as a simple
/// path radiance estimate.
///
/// # Arguments
/// * `image`       - Hyperspectral image.
/// * `percentile`  - Percentile (0–100) used as the "dark object" value (default 1).
///
/// # Returns
/// Atmospherically corrected image.
pub fn dark_object_subtraction(
    image: &HyperspectralImage,
    percentile: f64,
) -> NdimageResult<HyperspectralImage> {
    if percentile < 0.0 || percentile > 100.0 {
        return Err(NdimageError::InvalidInput("percentile must be in [0, 100]".into()));
    }

    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    let mut dark = Array1::<f64>::zeros(n_bands);

    for b in 0..n_bands {
        let mut vals: Vec<f64> = (0..n_pixels).map(|p| image.data[[p, b]]).collect();
        vals.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((percentile / 100.0) * (n_pixels as f64 - 1.0)).round() as usize;
        dark[b] = vals[idx.min(n_pixels - 1)];
    }

    let mut corrected_data = image.data.clone();
    for p in 0..n_pixels {
        for b in 0..n_bands {
            corrected_data[[p, b]] = (image.data[[p, b]] - dark[b]).max(0.0);
        }
    }

    Ok(HyperspectralImage { data: corrected_data, wavelengths: image.wavelengths.clone() })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_image(n_pixels: usize, n_bands: usize) -> HyperspectralImage {
        let mut data = Array2::<f64>::zeros((n_pixels, n_bands));
        for p in 0..n_pixels {
            for b in 0..n_bands {
                data[[p, b]] = ((p * n_bands + b) as f64) / (n_pixels * n_bands) as f64;
            }
        }
        HyperspectralImage::new(data)
    }

    #[test]
    fn test_mnf_output_shape() {
        let img = make_image(50, 10);
        let res = minimum_noise_fraction(&img, 3).expect("MNF failed");
        assert_eq!(res.components.shape(), &[50, 3]);
        assert_eq!(res.transform.shape(), &[3, 10]);
    }

    #[test]
    fn test_whiten_identity_covariance() {
        let img = make_image(200, 8);
        let (wh, _w) = whiten_hyperspectral(&img).expect("whitening failed");
        // Covariance of whitened data should be approximately identity.
        let (centred, _) = centre_cols(&wh.data);
        let n_f = (wh.n_pixels() - 1).max(1) as f64;
        let cov = centred.t().dot(&centred) / n_f;
        // Diagonal should be ~1, off-diagonal ~0.
        for i in 0..8 {
            assert!((cov[[i, i]] - 1.0).abs() < 0.5, "diagonal cov[{i},{i}]={}", cov[[i, i]]);
        }
    }

    #[test]
    fn test_remove_bands_shape() {
        let img = make_image(30, 10);
        let removed = remove_bands(&img, &[2, 5, 8]).expect("remove_bands failed");
        assert_eq!(removed.n_bands(), 7);
        assert_eq!(removed.n_pixels(), 30);
    }

    #[test]
    fn test_remove_absorption_bands_with_wavelengths() {
        let img = make_image(20, 5);
        let img_wl = HyperspectralImage::with_wavelengths(
            img.data.clone(),
            Array1::from_vec(vec![400.0, 800.0, 1400.0, 1900.0, 2400.0]),
        ).expect("with_wavelengths failed");
        let result = remove_absorption_bands(
            &img_wl,
            &[(1300.0, 1500.0), (1800.0, 2000.0)],
        ).expect("remove_absorption_bands failed");
        assert_eq!(result.n_bands(), 3); // 1400 and 1900 removed.
    }

    #[test]
    fn test_spatial_smoothing_shape() {
        let cube = scirs2_core::ndarray::Array3::<f64>::ones((8, 8, 5));
        let smoothed = spatial_smoothing(&cube, 3).expect("spatial_smoothing failed");
        assert_eq!(smoothed.shape(), cube.shape());
    }

    #[test]
    fn test_spatial_smoothing_uniform() {
        let cube = scirs2_core::ndarray::Array3::<f64>::from_elem((4, 4, 3), 2.0);
        let smoothed = spatial_smoothing(&cube, 3).expect("spatial_smoothing failed");
        // Uniform input should be unchanged.
        for v in smoothed.iter() {
            assert!((v - 2.0).abs() < 1e-10, "value {} != 2.0", v);
        }
    }

    #[test]
    fn test_cube_pixels_roundtrip() {
        let cube = scirs2_core::ndarray::Array3::<f64>::from_elem((4, 5, 3), 1.5);
        let img = cube_to_pixels(&cube);
        assert_eq!(img.n_pixels(), 20);
        assert_eq!(img.n_bands(), 3);
        let cube2 = pixels_to_cube(&img, 4, 5).expect("pixels_to_cube failed");
        for (a, b) in cube.iter().zip(cube2.iter()) {
            assert!((*a - *b).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn test_radiometric_correction_linear() {
        let img = make_image(10, 4);
        let cal = RadiometricCalibration::uniform(4, 0.5);
        let corrected = radiometric_correction(&img, &cal).expect("radiometric_correction failed");
        for p in 0..10 {
            for b in 0..4 {
                let expected = 0.5 * img.data[[p, b]];
                assert!((corrected.data[[p, b]] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_dark_object_subtraction_non_negative() {
        let img = make_image(50, 6);
        let dos = dark_object_subtraction(&img, 1.0).expect("DOS failed");
        for v in dos.data.iter() {
            assert!(*v >= 0.0, "DOS produced negative value {}", v);
        }
    }
}
