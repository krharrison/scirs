//! Spectral Unmixing for Hyperspectral Images
//!
//! Implements endmember extraction and abundance estimation algorithms
//! for hyperspectral imagery analysis.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use std::f64::consts::PI;

use crate::error::{NdimageError, NdimageResult};

// ─────────────────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────────────────

/// Hyperspectral image representation as a 2-D matrix.
///
/// Rows correspond to pixels, columns to spectral bands.
/// Shape: `[N_pixels, N_bands]`.
#[derive(Debug, Clone)]
pub struct HyperspectralImage {
    /// Pixel-by-band data matrix.
    pub data: Array2<f64>,
    /// Optional wavelength labels (nm) for each band.
    pub wavelengths: Option<Array1<f64>>,
}

impl HyperspectralImage {
    /// Create a new hyperspectral image from a data matrix.
    pub fn new(data: Array2<f64>) -> Self {
        Self { data, wavelengths: None }
    }

    /// Create with explicit wavelength labels.
    pub fn with_wavelengths(data: Array2<f64>, wavelengths: Array1<f64>) -> NdimageResult<Self> {
        let n_bands = data.ncols();
        if wavelengths.len() != n_bands {
            return Err(NdimageError::InvalidInput(format!(
                "wavelengths length {} != n_bands {}",
                wavelengths.len(),
                n_bands
            )));
        }
        Ok(Self { data, wavelengths: Some(wavelengths) })
    }

    /// Number of pixels.
    #[inline]
    pub fn n_pixels(&self) -> usize {
        self.data.nrows()
    }

    /// Number of spectral bands.
    #[inline]
    pub fn n_bands(&self) -> usize {
        self.data.ncols()
    }

    /// Compute per-band mean spectrum (1-D, length = n_bands).
    pub fn mean_spectrum(&self) -> Array1<f64> {
        self.data.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(self.n_bands()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal linear-algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the squared Euclidean norm of a 1-D slice.
#[inline]
fn norm_sq(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Compute the Euclidean norm of a 1-D slice.
#[inline]
fn norm(v: &[f64]) -> f64 {
    norm_sq(v).sqrt()
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Project `v` onto the orthogonal complement of the span of columns in `basis`
/// (each column is a unit vector already in orthonormal form).
fn project_out(v: &Array1<f64>, basis: &Array2<f64>, n_basis: usize) -> Array1<f64> {
    let mut result = v.clone();
    for k in 0..n_basis {
        let col = basis.column(k);
        let d = dot(result.as_slice().unwrap_or(&[]), col.to_owned().as_slice().unwrap_or(&[]));
        result = result - col.to_owned() * d;
    }
    result
}

/// Thin SVD: returns (U [m×r], S [r], Vt [r×n]) keeping `rank` singular values.
/// Uses the power-iteration randomised approach for moderate sizes.
fn thin_svd(a: &Array2<f64>, rank: usize) -> NdimageResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = (a.nrows(), a.ncols());
    let r = rank.min(m).min(n);

    // Build AAt or AtA depending on shape, then extract eigenvectors.
    // For hyperspectral data m >> n typically, so work with AtA (n×n).
    let ata = a.t().dot(a); // n×n

    // Power iteration to get top-r eigenvectors of AtA.
    let mut q = random_orthonormal_matrix(n, r);

    for _ in 0..20 {
        let z = ata.dot(&q);
        // QR of z
        q = qr_factorization(&z, r)?;
    }

    // B = A Q (m×r)
    let b = a.dot(&q);

    // SVD of small b (m×r) — use direct Gram-Schmidt on columns of b.
    let mut u = Array2::<f64>::zeros((m, r));
    let mut s = Array1::<f64>::zeros(r);
    let mut vt = Array2::<f64>::zeros((r, n));

    // QR of b to get U, then S = diagonal norms, Vt = Q' (of AtA projection).
    let mut b_ortho = b.clone();
    for k in 0..r {
        let col: Array1<f64> = b_ortho.column(k).to_owned();
        let n_col = norm(col.as_slice().unwrap_or(&[]));
        if n_col < 1e-14 {
            break;
        }
        let u_col = &col / n_col;
        for j in 0..m {
            u[[j, k]] = u_col[j];
        }
        s[k] = n_col;
        // Project out from remaining columns of b_ortho
        for j in (k + 1)..r {
            let c: Array1<f64> = b_ortho.column(j).to_owned();
            let proj = dot(c.as_slice().unwrap_or(&[]), u_col.as_slice().unwrap_or(&[]));
            for i in 0..m {
                b_ortho[[i, j]] -= proj * u_col[i];
            }
        }
        // vt row k = q column k normalised  (q already carries V directions)
        let q_col: Array1<f64> = q.column(k).to_owned();
        for j in 0..n {
            vt[[k, j]] = q_col[j];
        }
    }

    Ok((u, s, vt))
}

/// Deterministic near-random orthonormal matrix via Hadamard-like seed.
fn random_orthonormal_matrix(n: usize, r: usize) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((n, r));
    // Fill with pseudo-random values via a simple LCG.
    let mut state: u64 = 0x5EED_CAFE_DEAD_BEEF;
    for i in 0..n {
        for j in 0..r {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let f = ((state >> 11) as f64) / (1u64 << 53) as f64 - 0.5;
            m[[i, j]] = f;
        }
    }
    qr_factorization(&m, r).unwrap_or(m)
}

/// Thin QR factorization: returns the Q factor of shape (m × r).
fn qr_factorization(a: &Array2<f64>, r: usize) -> NdimageResult<Array2<f64>> {
    let m = a.nrows();
    let cols = r.min(a.ncols());
    let mut q = Array2::<f64>::zeros((m, cols));

    for k in 0..cols {
        let mut col: Array1<f64> = a.column(k).to_owned();
        // Subtract projections on already-orthonormalised columns.
        for j in 0..k {
            let q_j = q.column(j).to_owned();
            let p = dot(col.as_slice().unwrap_or(&[]), q_j.as_slice().unwrap_or(&[]));
            col = col - q_j * p;
        }
        let n = norm(col.as_slice().unwrap_or(&[]));
        if n < 1e-14 {
            // Near-zero column: insert a canonical basis vector.
            if k < m {
                q[[k, k]] = 1.0;
            }
            continue;
        }
        let unit = col / n;
        for i in 0..m {
            q[[i, k]] = unit[i];
        }
    }
    Ok(q)
}

/// Solve the least-squares system `A x = b` via normal equations (AtA x = Atb).
/// Returns `x` of length equal to `a.ncols()`.
fn lstsq(a: &Array2<f64>, b: &Array1<f64>) -> NdimageResult<Array1<f64>> {
    let ata = a.t().dot(a); // p×p
    let atb = a.t().dot(b); // p
    cholesky_solve(&ata, &atb)
}

/// Solve `A x = b` where `A` is symmetric positive-(semi)definite via Cholesky.
fn cholesky_solve(a: &Array2<f64>, b: &Array1<f64>) -> NdimageResult<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(NdimageError::InvalidInput("cholesky_solve: dimension mismatch".into()));
    }

    // Cholesky L L^T  (with Tikhonov regularisation if not pos-def).
    let reg = 1e-10 * a.diag().iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1e-10);
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let diag = s + reg;
                if diag <= 0.0 {
                    return Err(NdimageError::ComputationError("matrix not positive definite".into()));
                }
                l[[i, j]] = diag.sqrt();
            } else if l[[j, j]].abs() > 1e-15 {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }

    // Forward substitution L y = b
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * y[k];
        }
        if l[[i, i]].abs() > 1e-15 {
            y[i] = s / l[[i, i]];
        }
    }

    // Back substitution L^T x = y
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * x[k];
        }
        if l[[i, i]].abs() > 1e-15 {
            x[i] = s / l[[i, i]];
        }
    }
    Ok(x)
}

/// Solve for all abundance columns: each column of `b_mat` against matrix `a`.
/// Returns `X` of shape `[p, N]` where p = a.ncols(), N = b_mat.ncols().
fn lstsq_multi(a: &Array2<f64>, b_mat: &Array2<f64>) -> NdimageResult<Array2<f64>> {
    let p = a.ncols();
    let n_pix = b_mat.ncols();
    let mut x = Array2::<f64>::zeros((p, n_pix));
    let ata = a.t().dot(a);
    for i in 0..n_pix {
        let b_col = b_mat.column(i).to_owned();
        let atb = a.t().dot(&b_col);
        let xi = cholesky_solve(&ata, &atb)?;
        for j in 0..p {
            x[[j, i]] = xi[j];
        }
    }
    Ok(x)
}

/// Volume of the simplex spanned by columns of `e` (shape p×k endmembers).
/// Returns the unsigned volume scaled by 1/k!.
fn simplex_volume(e: &Array2<f64>) -> f64 {
    let (p, k) = (e.nrows(), e.ncols());
    if k == 0 || p == 0 {
        return 0.0;
    }
    // Build (k-1) × (k-1) difference matrix from first endmember.
    let e0 = e.column(0).to_owned();
    let dim = k - 1;
    if dim == 0 {
        return 1.0;
    }
    let mut diff = Array2::<f64>::zeros((p, dim));
    for j in 1..k {
        for i in 0..p {
            diff[[i, j - 1]] = e[[i, j]] - e0[i];
        }
    }
    // det(diff^T diff) approximation via Gram matrix.
    let gram = diff.t().dot(&diff); // dim×dim
    gram_determinant(&gram).abs().sqrt()
}

/// Determinant of a small square matrix via LU decomposition.
fn gram_determinant(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut lu = a.clone();
    let mut sign = 1.0_f64;

    for col in 0..n {
        // Partial pivot
        let mut max_val = lu[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if lu[[row, col]].abs() > max_val {
                max_val = lu[[row, col]].abs();
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..n {
                let tmp = lu[[col, j]];
                lu[[col, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
            sign = -sign;
        }
        if lu[[col, col]].abs() < 1e-14 {
            return 0.0;
        }
        for row in (col + 1)..n {
            let factor = lu[[row, col]] / lu[[col, col]];
            for j in col..n {
                let val = lu[[col, j]] * factor;
                lu[[row, j]] -= val;
            }
        }
    }
    let mut det = sign;
    for i in 0..n {
        det *= lu[[i, i]];
    }
    det
}

// ─────────────────────────────────────────────────────────────────────────────
// Endmember Extraction: VCA
// ─────────────────────────────────────────────────────────────────────────────

/// Vertex Component Analysis (VCA) endmember extraction.
///
/// Extracts `p` endmember spectra from a hyperspectral image using the VCA
/// algorithm (Nascimento & Bioucas-Dias, 2005).
///
/// # Arguments
/// * `image`   - Hyperspectral image (`[N_pixels, N_bands]`).
/// * `n_endmembers` - Number of endmembers to extract.
///
/// # Returns
/// `Array2<f64>` of shape `[N_bands, p]` — each column is an endmember spectrum.
pub fn vertex_component_analysis(
    image: &HyperspectralImage,
    n_endmembers: usize,
) -> NdimageResult<Array2<f64>> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    if n_endmembers == 0 || n_endmembers > n_pixels {
        return Err(NdimageError::InvalidInput(format!(
            "n_endmembers must be in 1..=n_pixels, got {}",
            n_endmembers
        )));
    }

    let data = &image.data; // [N_pixels, N_bands]

    // Step 1: Reduce dimensionality to (p-1) via PCA.
    let mean = image.mean_spectrum();
    let centered = data - &mean.view().insert_axis(Axis(0)); // [N, L]

    // Thin SVD of centered^T (L×N) up to rank p.
    let rank = n_endmembers.min(n_bands).min(n_pixels);
    let (_, _s, vt) = thin_svd(&centered, rank)?; // vt: [r, N]

    // Projected data: Ud = Vt * centered (each row is a basis component).
    // Shape: [r, N] . [N, L] → [r, L]  — but we want projected pixels [N, r].
    // Actually: center_t = centered.t() [L×N]; U [L×r]; coord = U^T center_t [r×N].
    // We already have vt from AtA decomp which gives right-singular vectors.
    // Reconstruct U from Vt: since vt is from AtA, vt rows are right-singular
    // vectors of centered, so projected coords = centered . Vt^T  [N×r].
    let projected = centered.dot(&vt.t()); // [N, r]

    // Pad to p dims by adding a constant row summed to 1.
    let p = n_endmembers;
    let r = projected.ncols();
    let dim = r.min(p);

    // Build working matrix R of shape [p, N]: each column is a projected pixel.
    let mut r_mat = Array2::<f64>::zeros((p, n_pixels));
    for i in 0..n_pixels {
        for j in 0..dim {
            r_mat[[j, i]] = projected[[i, j]];
        }
        // Last row is constant to give an affine embedding.
        r_mat[[p - 1, i]] = 1.0;
    }

    // VCA main loop: iteratively find vertices.
    let mut endmember_cols = Array2::<f64>::zeros((n_bands, p));
    let mut a = Array2::<f64>::zeros((p, p));
    // Initialise A with random column in the data.
    let init_idx = n_pixels / 2;
    for k in 0..p {
        a[[k, 0]] = r_mat[[k, init_idx]];
    }

    let mut selected_indices = vec![0usize; p];

    for i in 0..p {
        // Project each column of R_mat orthogonal to span(A) and find argmax norm.
        // f = (I - A A^+) r_j  where A^+ is pseudoinverse.
        // Equivalent: find pixel farthest from span of current endmembers.

        // Build orthonormal basis from columns of A[:, 0..i].
        let a_sub = a.slice(s![.., 0..i.max(1)]).to_owned();
        let a_orth = qr_factorization(&a_sub, i.max(1))?;

        let mut max_val = -1.0_f64;
        let mut best_idx = 0usize;

        for j in 0..n_pixels {
            let col = r_mat.column(j).to_owned();
            let proj = project_out(&col, &a_orth, i);
            let v = norm(proj.as_slice().unwrap_or(&[]));
            if v > max_val {
                max_val = v;
                best_idx = j;
            }
        }

        selected_indices[i] = best_idx;
        let pixel_col = r_mat.column(best_idx).to_owned();
        for k in 0..p {
            a[[k, i]] = pixel_col[k];
        }

        // Store the original spectrum.
        let spectrum = data.row(best_idx).to_owned();
        for k in 0..n_bands {
            endmember_cols[[k, i]] = spectrum[k];
        }
    }

    Ok(endmember_cols)
}

// ─────────────────────────────────────────────────────────────────────────────
// Endmember Extraction: N-FINDR
// ─────────────────────────────────────────────────────────────────────────────

/// N-FINDR algorithm for endmember extraction (Winter 1999).
///
/// Iteratively replaces endmembers to maximise the simplex volume enclosed
/// by the selected spectra. Guaranteed to converge but sensitive to
/// initialisation; multiple restarts are performed automatically.
///
/// # Arguments
/// * `image`        - Hyperspectral image `[N_pixels, N_bands]`.
/// * `n_endmembers` - Number of endmembers `p`.
/// * `max_iter`     - Maximum number of replacement iterations per restart.
/// * `n_restarts`   - Number of random-initialisation restarts.
///
/// # Returns
/// `Array2<f64>` of shape `[N_bands, p]`.
pub fn nfindr(
    image: &HyperspectralImage,
    n_endmembers: usize,
    max_iter: usize,
    n_restarts: usize,
) -> NdimageResult<Array2<f64>> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();

    if n_endmembers < 2 {
        return Err(NdimageError::InvalidInput("n_endmembers must be >= 2 for N-FINDR".into()));
    }
    if n_endmembers > n_pixels {
        return Err(NdimageError::InvalidInput(format!(
            "n_endmembers {} > n_pixels {}",
            n_endmembers, n_pixels
        )));
    }

    let data = &image.data; // [N_pixels, N_bands]

    // Dimensionality reduction to (p-1) dims via PCA to make volume computation tractable.
    let p = n_endmembers;
    let reduce_dim = (p - 1).min(n_bands);

    let mean = image.mean_spectrum();
    let centered = data - &mean.view().insert_axis(Axis(0));

    let (_u, _s, vt) = thin_svd(&centered, reduce_dim)?; // vt: [reduce_dim, N_bands]
    let projected = centered.dot(&vt.t()); // [N_pixels, reduce_dim]

    // Helper: build endmember matrix from index set.
    let build_e_matrix = |indices: &[usize]| -> Array2<f64> {
        let mut e = Array2::<f64>::zeros((reduce_dim, p));
        for (j, &idx) in indices.iter().enumerate() {
            for k in 0..reduce_dim {
                e[[k, j]] = projected[[idx, k]];
            }
        }
        e
    };

    let mut best_vol = -1.0_f64;
    let mut best_indices = vec![0usize; p];

    // Simple deterministic index generator for reproducible restarts.
    let mut lcg_state: u64 = 0xDEAD_BEEF_1337_CAFE;
    let lcg_next = |state: &mut u64| -> usize {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as usize) % n_pixels
    };

    for _restart in 0..n_restarts.max(1) {
        // Initialise with well-spread pixel indices.
        let mut indices: Vec<usize> = if _restart == 0 {
            // Spread uniformly.
            (0..p).map(|i| (i * n_pixels / p)).collect()
        } else {
            // Random initialisation.
            let mut idx_set: Vec<usize> = Vec::with_capacity(p);
            while idx_set.len() < p {
                let candidate = lcg_next(&mut lcg_state);
                if !idx_set.contains(&candidate) {
                    idx_set.push(candidate);
                }
            }
            idx_set
        };

        let mut current_vol = simplex_volume(&build_e_matrix(&indices));

        for _iter in 0..max_iter {
            let mut improved = false;
            for i in 0..p {
                let old_idx = indices[i];
                for j in 0..n_pixels {
                    if indices.contains(&j) && j != old_idx {
                        continue;
                    }
                    indices[i] = j;
                    let vol = simplex_volume(&build_e_matrix(&indices));
                    if vol > current_vol {
                        current_vol = vol;
                        improved = true;
                    } else {
                        indices[i] = old_idx;
                    }
                }
            }
            if !improved {
                break;
            }
        }

        if current_vol > best_vol {
            best_vol = current_vol;
            best_indices.clone_from(&indices);
        }
    }

    // Retrieve original spectra.
    let mut endmembers = Array2::<f64>::zeros((n_bands, p));
    for (j, &idx) in best_indices.iter().enumerate() {
        let spectrum = data.row(idx).to_owned();
        for k in 0..n_bands {
            endmembers[[k, j]] = spectrum[k];
        }
    }
    Ok(endmembers)
}

// ─────────────────────────────────────────────────────────────────────────────
// Endmember Extraction: SISAL (simplified)
// ─────────────────────────────────────────────────────────────────────────────

/// SISAL — Simplex Identification via Split Augmented Lagrangian (simplified).
///
/// A convex-geometry approach that finds the minimum-volume simplex
/// enclosing the data, allowing for noise (data may lie outside the simplex).
/// This implementation uses an alternating-direction iterative scheme.
///
/// Reference: Bioucas-Dias (2009), adapted to a tractable iterative form.
///
/// # Arguments
/// * `image`        - Hyperspectral image `[N_pixels, N_bands]`.
/// * `n_endmembers` - Number of endmembers `p`.
/// * `tau`          - Regularisation weight for data fit (default ≈ 1e-4).
/// * `max_iter`     - Maximum outer iterations.
///
/// # Returns
/// `Array2<f64>` of shape `[N_bands, p]`.
pub fn sisal(
    image: &HyperspectralImage,
    n_endmembers: usize,
    tau: f64,
    max_iter: usize,
) -> NdimageResult<Array2<f64>> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();

    if n_endmembers < 2 {
        return Err(NdimageError::InvalidInput("n_endmembers must be >= 2".into()));
    }
    if tau <= 0.0 {
        return Err(NdimageError::InvalidInput("tau must be positive".into()));
    }

    let p = n_endmembers;
    let data = &image.data;

    // Reduce to (p-1)-dimensional subspace.
    let dim = (p - 1).min(n_bands);
    let mean = image.mean_spectrum();
    let centered = data - &mean.view().insert_axis(Axis(0));

    let (_u, _s, vt) = thin_svd(&centered, dim)?;
    // projected: [N_pixels, dim]
    let projected = centered.dot(&vt.t());

    // Initialise endmembers with VCA in projected space.
    let init_img = HyperspectralImage::new(projected.clone());
    let e_init = vertex_component_analysis(&init_img, p)?; // [dim, p]

    // SISAL alternating-direction iterations.
    // Minimise:  0.5 * ||Y - E A||^2 + tau * log det(E)
    // where Y = projected^T [dim × N], A [p × N] are the abundances.
    // We alternate between updating A (closed-form) and E (gradient step).

    let y = projected.t().to_owned(); // [dim, N]

    let mut e = e_init.clone(); // [dim, p]

    // Pre-compute regularisation for numerical stability.
    let mu = tau / (p as f64);

    for _iter in 0..max_iter {
        // ── Update A ── closed-form unconstrained LS.
        // A = (E^T E)^{-1} E^T Y  for each column of Y.
        let ete = e.t().dot(&e); // [p, p]
        let ety = e.t().dot(&y); // [p, N]

        // Solve ete * a = ety column by column.
        let mut a = Array2::<f64>::zeros((p, n_pixels));
        for i in 0..n_pixels {
            let rhs = ety.column(i).to_owned();
            match cholesky_solve(&ete, &rhs) {
                Ok(ai) => {
                    for j in 0..p {
                        a[[j, i]] = ai[j];
                    }
                }
                Err(_) => {
                    // Fallback: copy rhs proportionally.
                    let s: f64 = rhs.iter().map(|x| x.abs()).sum::<f64>().max(1e-10);
                    for j in 0..p {
                        a[[j, i]] = rhs[j] / s;
                    }
                }
            }
        }

        // ── Update E ── gradient + log-det term.
        // Gradient w.r.t. E: -(Y - E A) A^T + mu * E^{-T}
        let residual = &y - &e.dot(&a); // [dim, N]
        let grad_data = -residual.dot(&a.t()); // [dim, p]

        // Approximate E^{-T} gradient contribution via E (E^T E)^{-1}.
        // This is the Moore–Penrose right pseudo-inverse.
        let ata = e.t().dot(&e); // [p, p]
        let pinv_e = match cholesky_solve(&ata, &Array1::from_vec(vec![1.0; p])) {
            Ok(_) => {
                // Build full pseudo-inv: pinv = E (EtE)^{-1}  [dim × p]
                let mut pi = Array2::<f64>::zeros((dim, p));
                for k in 0..p {
                    let mut ek = Array1::<f64>::zeros(p);
                    ek[k] = 1.0;
                    if let Ok(sol) = cholesky_solve(&ata, &ek) {
                        let ep_col = e.dot(&sol); // [dim]
                        for i in 0..dim {
                            pi[[i, k]] = ep_col[i];
                        }
                    }
                }
                pi
            }
            Err(_) => Array2::<f64>::zeros((dim, p)),
        };

        let step = 1e-3 / (1.0 + _iter as f64 * 0.1);
        e = e - (grad_data + &pinv_e * mu) * step;
    }

    // Map reduced endmembers back to original spectral space.
    // E_original = mean + E_reduced @ Vt   [p, L]
    let e_original_t = e.t().dot(&vt); // [p, L]
    let mut endmembers = Array2::<f64>::zeros((n_bands, p));
    for j in 0..p {
        for k in 0..n_bands {
            endmembers[[k, j]] = e_original_t[[j, k]] + mean[k];
        }
    }
    Ok(endmembers)
}

// ─────────────────────────────────────────────────────────────────────────────
// Abundance estimation: UCLS
// ─────────────────────────────────────────────────────────────────────────────

/// Unconstrained Least Squares (UCLS) abundance estimation.
///
/// Solves `y = E a` in the LS sense for each pixel, with no constraints.
///
/// # Arguments
/// * `image`      - Hyperspectral image `[N_pixels, N_bands]`.
/// * `endmembers` - Endmember matrix `[N_bands, p]`.
///
/// # Returns
/// Abundance matrix `[N_pixels, p]`.
pub fn abundance_estimation_ucls(
    image: &HyperspectralImage,
    endmembers: &Array2<f64>,
) -> NdimageResult<Array2<f64>> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    let p = endmembers.ncols();

    if endmembers.nrows() != n_bands {
        return Err(NdimageError::InvalidInput(format!(
            "endmembers.nrows() {} != n_bands {}",
            endmembers.nrows(),
            n_bands
        )));
    }

    // Solve E a = y (pixel by pixel).
    // E: [L × p],  y: [L]   →  a: [p]
    let e = endmembers; // [L, p]
    let data_t = image.data.t().to_owned(); // [L, N]

    // Precompute (EtE)^{-1} Et once.
    let ete = e.t().dot(e); // [p, p]
    let mut abundances = Array2::<f64>::zeros((n_pixels, p));

    for i in 0..n_pixels {
        let y_i = data_t.column(i).to_owned();
        let ety_i = e.t().dot(&y_i); // [p]
        let a_i = cholesky_solve(&ete, &ety_i)?;
        for j in 0..p {
            abundances[[i, j]] = a_i[j];
        }
    }
    Ok(abundances)
}

// ─────────────────────────────────────────────────────────────────────────────
// Abundance estimation: NCLS
// ─────────────────────────────────────────────────────────────────────────────

/// Non-negative Constrained Least Squares (NCLS) abundance estimation.
///
/// Enforces the abundance non-negativity constraint (ANC): `a_k >= 0`.
/// Uses the active-set (NNLS) method (Lawson & Hanson 1974).
///
/// # Arguments
/// * `image`      - Hyperspectral image `[N_pixels, N_bands]`.
/// * `endmembers` - `[N_bands, p]`.
///
/// # Returns
/// Abundance matrix `[N_pixels, p]` with all non-negative entries.
pub fn abundance_estimation_ncls(
    image: &HyperspectralImage,
    endmembers: &Array2<f64>,
) -> NdimageResult<Array2<f64>> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    let p = endmembers.ncols();

    if endmembers.nrows() != n_bands {
        return Err(NdimageError::InvalidInput(format!(
            "endmembers.nrows() {} != n_bands {}",
            endmembers.nrows(),
            n_bands
        )));
    }

    let e = endmembers;
    let ete = e.t().dot(e); // [p, p]

    let mut abundances = Array2::<f64>::zeros((n_pixels, p));
    for i in 0..n_pixels {
        let y_i = image.data.row(i).to_owned();
        let ety_i = e.t().dot(&y_i);
        let a_i = nnls_active_set(&ete, &ety_i, 200)?;
        for j in 0..p {
            abundances[[i, j]] = a_i[j];
        }
    }
    Ok(abundances)
}

/// Non-negative Least Squares via the active-set algorithm.
/// Solves `min ||Ax - b||^2` s.t. `x >= 0`, given `A^T A` and `A^T b`.
fn nnls_active_set(ata: &Array2<f64>, atb: &Array1<f64>, max_iter: usize) -> NdimageResult<Array1<f64>> {
    let p = ata.nrows();
    let mut x = Array1::<f64>::zeros(p);
    // Passive set: indices where x > 0.
    let mut passive = vec![false; p];

    for _outer in 0..max_iter {
        // Compute gradient of 0.5 xT AtA x - xT Atb: g = AtA x - Atb
        let g = ata.dot(&x) - atb;

        // Find index not in passive with most negative gradient.
        let mut t_idx = p; // sentinel
        let mut t_val = -1e-10_f64;
        for j in 0..p {
            if !passive[j] && -g[j] > t_val {
                t_val = -g[j];
                t_idx = j;
            }
        }
        if t_idx == p {
            break; // KKT satisfied.
        }
        passive[t_idx] = true;

        loop {
            // Solve unconstrained LS on passive set.
            let passive_indices: Vec<usize> = (0..p).filter(|&j| passive[j]).collect();
            let k = passive_indices.len();
            let mut ata_p = Array2::<f64>::zeros((k, k));
            let mut atb_p = Array1::<f64>::zeros(k);
            for (a, &pa) in passive_indices.iter().enumerate() {
                atb_p[a] = atb[pa];
                for (b, &pb) in passive_indices.iter().enumerate() {
                    ata_p[[a, b]] = ata[[pa, pb]];
                }
            }
            let s_p = cholesky_solve(&ata_p, &atb_p)?;

            // If all passive-set values are positive, accept.
            if s_p.iter().all(|&v| v > 0.0) {
                for (a, &pa) in passive_indices.iter().enumerate() {
                    x[pa] = s_p[a];
                }
                break;
            }

            // Find step alpha to bring a negative s_p[a] to zero.
            let mut alpha = f64::INFINITY;
            for (a, &pa) in passive_indices.iter().enumerate() {
                if s_p[a] <= 0.0 {
                    let ratio = x[pa] / (x[pa] - s_p[a]);
                    if ratio < alpha {
                        alpha = ratio;
                    }
                }
            }

            // Update x and deactivate those reaching zero.
            for (a, &pa) in passive_indices.iter().enumerate() {
                x[pa] += alpha * (s_p[a] - x[pa]);
                if x[pa] < 1e-12 {
                    x[pa] = 0.0;
                    passive[pa] = false;
                }
            }
        }
    }

    // Clamp residual negatives from floating-point drift.
    x.iter_mut().for_each(|v| {
        if *v < 0.0 {
            *v = 0.0;
        }
    });
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Abundance estimation: FCLS
// ─────────────────────────────────────────────────────────────────────────────

/// Fully Constrained Least Squares (FCLS) abundance estimation.
///
/// Enforces both the abundance non-negativity constraint (ANC: `a_k >= 0`) and
/// the abundance sum-to-one constraint (ASC: `sum(a_k) = 1`).
///
/// The ASC is incorporated by augmenting the endmember and pixel matrices with
/// a scaled row of ones (Chang & Heinz 2002).
///
/// # Arguments
/// * `image`      - Hyperspectral image `[N_pixels, N_bands]`.
/// * `endmembers` - `[N_bands, p]`.
/// * `delta`      - ASC enforcement weight (default ≈ `10 * max(E)`).
///
/// # Returns
/// Abundance matrix `[N_pixels, p]` satisfying ANC + ASC.
pub fn abundance_estimation_fcls(
    image: &HyperspectralImage,
    endmembers: &Array2<f64>,
    delta: f64,
) -> NdimageResult<Array2<f64>> {
    let n_pixels = image.n_pixels();
    let n_bands = image.n_bands();
    let p = endmembers.ncols();

    if endmembers.nrows() != n_bands {
        return Err(NdimageError::InvalidInput(format!(
            "endmembers.nrows() {} != n_bands {}",
            endmembers.nrows(),
            n_bands
        )));
    }
    if delta <= 0.0 {
        return Err(NdimageError::InvalidInput("delta must be positive".into()));
    }

    // Augment E with a row of delta * ones to enforce ASC.
    let mut e_aug = Array2::<f64>::zeros((n_bands + 1, p));
    for k in 0..n_bands {
        for j in 0..p {
            e_aug[[k, j]] = endmembers[[k, j]];
        }
    }
    for j in 0..p {
        e_aug[[n_bands, j]] = delta;
    }

    let ete_aug = e_aug.t().dot(&e_aug); // [p, p]

    let mut abundances = Array2::<f64>::zeros((n_pixels, p));
    for i in 0..n_pixels {
        let y_i = image.data.row(i).to_owned();
        // Augmented pixel: append delta (since ASC target = 1).
        let mut y_aug = Array1::<f64>::zeros(n_bands + 1);
        for k in 0..n_bands {
            y_aug[k] = y_i[k];
        }
        y_aug[n_bands] = delta;

        let ety_aug = e_aug.t().dot(&y_aug); // [p]
        let a_i = nnls_active_set(&ete_aug, &ety_aug, 300)?;

        // Normalise to sum to 1 if strictly positive (avoid division by zero).
        let s: f64 = a_i.iter().sum();
        for j in 0..p {
            abundances[[i, j]] = if s > 1e-12 { a_i[j] / s } else { 1.0 / p as f64 };
        }
    }
    Ok(abundances)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a synthetic 2-endmember scene.
    fn make_two_endmember_scene(n_pixels: usize, n_bands: usize) -> (HyperspectralImage, Array2<f64>) {
        let mut data = Array2::<f64>::zeros((n_pixels, n_bands));
        let e1: Vec<f64> = (0..n_bands).map(|b| (b as f64 / n_bands as f64)).collect();
        let e2: Vec<f64> = (0..n_bands).map(|b| 1.0 - b as f64 / n_bands as f64).collect();

        let mut endmembers = Array2::<f64>::zeros((n_bands, 2));
        for b in 0..n_bands {
            endmembers[[b, 0]] = e1[b];
            endmembers[[b, 1]] = e2[b];
        }

        let mut lcg: u64 = 0xABCDEF_1234;
        for i in 0..n_pixels {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            let alpha = ((lcg >> 33) as f64) / u32::MAX as f64;
            for b in 0..n_bands {
                data[[i, b]] = alpha * e1[b] + (1.0 - alpha) * e2[b];
            }
        }
        (HyperspectralImage::new(data), endmembers)
    }

    #[test]
    fn test_vca_returns_correct_shape() {
        let (img, _) = make_two_endmember_scene(100, 20);
        let e = vertex_component_analysis(&img, 2).expect("VCA failed");
        assert_eq!(e.shape(), &[20, 2]);
    }

    #[test]
    fn test_nfindr_returns_correct_shape() {
        let (img, _) = make_two_endmember_scene(50, 10);
        let e = nfindr(&img, 2, 20, 2).expect("N-FINDR failed");
        assert_eq!(e.shape(), &[10, 2]);
    }

    #[test]
    fn test_sisal_returns_correct_shape() {
        let (img, _) = make_two_endmember_scene(60, 12);
        let e = sisal(&img, 2, 1e-4, 30).expect("SISAL failed");
        assert_eq!(e.shape(), &[12, 2]);
    }

    #[test]
    fn test_ucls_shape_and_values() {
        let (img, endmembers) = make_two_endmember_scene(40, 10);
        let a = abundance_estimation_ucls(&img, &endmembers).expect("UCLS failed");
        assert_eq!(a.shape(), &[40, 2]);
        // For the two-endmember scene the sum of abundances should be ~ 1.
        for i in 0..40 {
            let s = a[[i, 0]] + a[[i, 1]];
            assert!((s - 1.0).abs() < 0.05, "UCLS sum={s} for pixel {i}");
        }
    }

    #[test]
    fn test_ncls_non_negative() {
        let (img, endmembers) = make_two_endmember_scene(40, 10);
        let a = abundance_estimation_ncls(&img, &endmembers).expect("NCLS failed");
        for &v in a.iter() {
            assert!(v >= 0.0, "NCLS produced negative abundance {v}");
        }
    }

    #[test]
    fn test_fcls_constraints() {
        let (img, endmembers) = make_two_endmember_scene(40, 10);
        let max_val = endmembers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let delta = 10.0 * max_val.max(1.0);
        let a = abundance_estimation_fcls(&img, &endmembers, delta).expect("FCLS failed");
        for i in 0..40 {
            let s: f64 = (0..2).map(|j| a[[i, j]]).sum();
            assert!((s - 1.0).abs() < 0.05, "FCLS ASC violated: sum={s}");
            for j in 0..2 {
                assert!(a[[i, j]] >= -1e-9, "FCLS ANC violated: a={}", a[[i, j]]);
            }
        }
    }

    #[test]
    fn test_nnls_all_zero_rhs() {
        let ata = Array2::<f64>::eye(3);
        let atb = Array1::<f64>::zeros(3);
        let x = nnls_active_set(&ata, &atb, 50).expect("NNLS failed");
        for &v in x.iter() {
            assert!(v >= 0.0);
        }
    }
}
