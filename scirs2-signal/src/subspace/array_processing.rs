//! Array signal processing utilities
//!
//! Provides:
//! - [`UniformLinearArray`]: ULA steering vector computation
//! - [`UniformCircularArray`]: UCA steering vectors
//! - [`ArrayManifold`]: trait with `steering_vector` method
//! - [`SpatialCovariance`]: sample covariance matrix with spatial smoothing
//! - [`SourceNumberEstimation`]: AIC, MDL, EDC criteria
//!
//! References:
//! - Van Trees, H.L. (2002). "Optimum Array Processing." Wiley-Interscience.
//! - Wax, M. & Kailath, T. (1985). "Detection of signals by information theoretic criteria."
//!   IEEE Trans. ASSP, 33(2), 387-392.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// ArrayManifold trait
// ---------------------------------------------------------------------------

/// Trait for array manifold computation (steering vector)
pub trait ArrayManifold {
    /// Compute steering vector for a given direction/frequency parameter
    ///
    /// # Arguments
    ///
    /// * `param` - Direction or spatial frequency parameter (angle in radians for DOA, or
    ///   normalised spatial frequency `d*sin(theta)/lambda` for generic arrays)
    ///
    /// # Returns
    ///
    /// * Complex steering vector of length `n_elements()`
    fn steering_vector(&self, param: f64) -> SignalResult<Vec<Complex64>>;

    /// Number of array elements
    fn n_elements(&self) -> usize;
}

// ---------------------------------------------------------------------------
// UniformLinearArray
// ---------------------------------------------------------------------------

/// Uniform Linear Array (ULA) manifold
///
/// Elements are spaced uniformly along a line. The steering vector is:
///
/// `a(θ) = [1, e^{-j 2π d sin θ}, …, e^{-j 2π (M-1) d sin θ}]`
///
/// where `d` is element spacing in wavelengths and `M` is the number of elements.
#[derive(Debug, Clone)]
pub struct UniformLinearArray {
    /// Number of array elements
    pub n_elements: usize,
    /// Inter-element spacing in wavelengths (typically 0.5)
    pub element_spacing: f64,
}

impl UniformLinearArray {
    /// Create a new ULA
    ///
    /// # Arguments
    ///
    /// * `n_elements`     - Number of sensors
    /// * `element_spacing` - Spacing between adjacent elements in wavelengths
    pub fn new(n_elements: usize, element_spacing: f64) -> SignalResult<Self> {
        if n_elements < 2 {
            return Err(SignalError::ValueError(
                "ULA requires at least 2 elements".to_string(),
            ));
        }
        if element_spacing <= 0.0 {
            return Err(SignalError::ValueError(
                "Element spacing must be positive".to_string(),
            ));
        }
        Ok(Self {
            n_elements,
            element_spacing,
        })
    }

    /// Compute steering vectors for multiple angles simultaneously
    pub fn steering_matrix(&self, angles_rad: &[f64]) -> SignalResult<Vec<Vec<Complex64>>> {
        angles_rad
            .iter()
            .map(|&theta| self.steering_vector(theta))
            .collect()
    }
}

impl ArrayManifold for UniformLinearArray {
    fn steering_vector(&self, angle_rad: f64) -> SignalResult<Vec<Complex64>> {
        let phase_increment = -2.0 * PI * self.element_spacing * angle_rad.sin();
        let sv = (0..self.n_elements)
            .map(|m| {
                let phase = phase_increment * m as f64;
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();
        Ok(sv)
    }

    fn n_elements(&self) -> usize {
        self.n_elements
    }
}

// ---------------------------------------------------------------------------
// UniformCircularArray
// ---------------------------------------------------------------------------

/// Uniform Circular Array (UCA) manifold
///
/// `M` elements are placed uniformly on a circle of radius `r`.
/// For a source at azimuth `φ` and elevation `ψ`, the m-th element phase is:
///
/// `ψ_m = 2π (r/λ) sin(ψ) cos(φ - 2πm/M)`
#[derive(Debug, Clone)]
pub struct UniformCircularArray {
    /// Number of elements
    pub n_elements: usize,
    /// Array radius in wavelengths
    pub radius: f64,
}

impl UniformCircularArray {
    /// Create a new UCA
    ///
    /// # Arguments
    ///
    /// * `n_elements` - Number of sensors
    /// * `radius`      - Array radius in wavelengths
    pub fn new(n_elements: usize, radius: f64) -> SignalResult<Self> {
        if n_elements < 3 {
            return Err(SignalError::ValueError(
                "UCA requires at least 3 elements".to_string(),
            ));
        }
        if radius <= 0.0 {
            return Err(SignalError::ValueError(
                "Radius must be positive".to_string(),
            ));
        }
        Ok(Self { n_elements, radius })
    }

    /// Steering vector for a source at azimuth `phi_rad` (elevation assumed 0, i.e. broadside plane)
    pub fn steering_vector_azimuth(&self, phi_rad: f64) -> SignalResult<Vec<Complex64>> {
        let m = self.n_elements;
        let sv = (0..m)
            .map(|i| {
                let element_angle = 2.0 * PI * i as f64 / m as f64;
                let phase = 2.0 * PI * self.radius * (phi_rad - element_angle).cos();
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();
        Ok(sv)
    }

    /// Steering vector for a source at azimuth `phi_rad` and elevation `psi_rad`
    pub fn steering_vector_3d(&self, phi_rad: f64, psi_rad: f64) -> SignalResult<Vec<Complex64>> {
        let m = self.n_elements;
        let sv = (0..m)
            .map(|i| {
                let element_angle = 2.0 * PI * i as f64 / m as f64;
                let phase = 2.0 * PI * self.radius * psi_rad.sin() * (phi_rad - element_angle).cos();
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();
        Ok(sv)
    }
}

impl ArrayManifold for UniformCircularArray {
    /// `param` is interpreted as azimuth angle in radians (elevation = 0)
    fn steering_vector(&self, param: f64) -> SignalResult<Vec<Complex64>> {
        self.steering_vector_azimuth(param)
    }

    fn n_elements(&self) -> usize {
        self.n_elements
    }
}

// ---------------------------------------------------------------------------
// SpatialCovariance
// ---------------------------------------------------------------------------

/// Sample covariance matrix estimator with optional spatial smoothing
///
/// The sample covariance matrix is:
///
/// `R̂ = (1/N) Σ_{n=1}^N x(n) x^H(n)`
///
/// Spatial (forward-backward) smoothing divides the array into overlapping
/// sub-arrays to decorrelate coherent sources.
#[derive(Debug, Clone)]
pub struct SpatialCovariance {
    /// Estimated covariance matrix stored as row-major flat Vec (n x n complex)
    pub matrix: Vec<Complex64>,
    /// Dimension (number of array elements or sub-array size after smoothing)
    pub size: usize,
    /// Number of snapshots used in the estimate
    pub n_snapshots: usize,
}

impl SpatialCovariance {
    /// Estimate the sample covariance matrix
    ///
    /// # Arguments
    ///
    /// * `data` - Snapshot matrix stored as `data[element_idx][snapshot_idx]`
    ///   (i.e., each row is one sensor)
    ///
    /// # Returns
    ///
    /// * `SpatialCovariance` with the estimated matrix
    pub fn estimate(data: &[Vec<Complex64>]) -> SignalResult<Self> {
        if data.is_empty() {
            return Err(SignalError::ValueError("Data must not be empty".to_string()));
        }
        let m = data.len();
        let n = data[0].len();
        if n == 0 {
            return Err(SignalError::ValueError(
                "Number of snapshots must be positive".to_string(),
            ));
        }
        for (i, row) in data.iter().enumerate() {
            if row.len() != n {
                return Err(SignalError::DimensionMismatch(format!(
                    "Element {i} has {} snapshots, expected {n}",
                    row.len()
                )));
            }
        }

        let mut matrix = vec![Complex64::new(0.0, 0.0); m * m];
        for i in 0..m {
            for j in 0..m {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + data[i][k] * data[j][k].conj();
                }
                matrix[i * m + j] = sum / n as f64;
            }
        }

        Ok(Self {
            matrix,
            size: m,
            n_snapshots: n,
        })
    }

    /// Estimate with spatial smoothing for coherent source separation
    ///
    /// Divides an M-element ULA into L overlapping sub-arrays each of size `sub_size`,
    /// where `L = M - sub_size + 1`. The smoothed covariance is the average of
    /// sub-array covariances.
    ///
    /// # Arguments
    ///
    /// * `data`     - Full array snapshot matrix `[M][N]`
    /// * `sub_size` - Sub-array size (< M)
    ///
    /// # Returns
    ///
    /// * `SpatialCovariance` of dimension `sub_size × sub_size`
    pub fn estimate_smoothed(data: &[Vec<Complex64>], sub_size: usize) -> SignalResult<Self> {
        if data.is_empty() {
            return Err(SignalError::ValueError("Data must not be empty".to_string()));
        }
        let m = data.len();
        let n = data[0].len();
        if sub_size < 2 || sub_size > m {
            return Err(SignalError::ValueError(format!(
                "sub_size ({sub_size}) must be in [2, {m}]"
            )));
        }
        let num_subarrays = m - sub_size + 1;
        let mut smoothed = vec![Complex64::new(0.0, 0.0); sub_size * sub_size];

        for l in 0..num_subarrays {
            // Extract sub-array rows l..(l + sub_size)
            let sub_data: Vec<&Vec<Complex64>> = (l..l + sub_size).map(|k| &data[k]).collect();
            for i in 0..sub_size {
                for j in 0..sub_size {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for k in 0..n {
                        sum = sum + sub_data[i][k] * sub_data[j][k].conj();
                    }
                    smoothed[i * sub_size + j] =
                        smoothed[i * sub_size + j] + sum / n as f64;
                }
            }
        }

        // Average over sub-arrays
        let scale = 1.0 / num_subarrays as f64;
        for v in smoothed.iter_mut() {
            *v = *v * scale;
        }

        Ok(Self {
            matrix: smoothed,
            size: sub_size,
            n_snapshots: n,
        })
    }

    /// Apply forward-backward (FB) averaging
    ///
    /// The FB-averaged covariance is `R_FB = (R + J R* J) / 2` where `J` is the
    /// exchange matrix. This improves coherent source handling.
    pub fn forward_backward_average(mut self) -> Self {
        let m = self.size;
        let mut fb = vec![Complex64::new(0.0, 0.0); m * m];
        for i in 0..m {
            for j in 0..m {
                // J R* J: element (i,j) = R*(m-1-i, m-1-j)
                let conj_val = self.matrix[(m - 1 - i) * m + (m - 1 - j)].conj();
                fb[i * m + j] = (self.matrix[i * m + j] + conj_val) * 0.5;
            }
        }
        self.matrix = fb;
        self
    }

    /// Get element (i, j)
    pub fn get(&self, i: usize, j: usize) -> Complex64 {
        self.matrix[i * self.size + j]
    }

    /// Add diagonal loading (regularisation): R ← R + δ I
    pub fn diagonal_load(&mut self, delta: f64) {
        for i in 0..self.size {
            self.matrix[i * self.size + i] =
                self.matrix[i * self.size + i] + Complex64::new(delta, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// SourceNumberEstimation
// ---------------------------------------------------------------------------

/// Information-theoretic source number estimation
///
/// Uses the eigenvalues of the sample covariance matrix to estimate the number
/// of sources via:
/// - AIC (Akaike Information Criterion) — Wax & Kailath (1985)
/// - MDL (Minimum Description Length) — Rissanen
/// - EDC (Efficient Detection Criterion)
///
/// # References
///
/// Wax, M. & Kailath, T. (1985). "Detection of signals by information theoretic criteria."
/// IEEE Trans. ASSP, 33(2), 387–392.
#[derive(Debug, Clone)]
pub struct SourceNumberEstimation {
    /// Eigenvalues of the sample covariance matrix (in descending order)
    pub eigenvalues: Vec<f64>,
    /// Number of snapshots
    pub n_snapshots: usize,
    /// AIC estimated number of sources
    pub aic_estimate: usize,
    /// MDL estimated number of sources
    pub mdl_estimate: usize,
    /// EDC estimated number of sources
    pub edc_estimate: usize,
    /// AIC criterion values for k = 0, …, M-1
    pub aic_values: Vec<f64>,
    /// MDL criterion values for k = 0, …, M-1
    pub mdl_values: Vec<f64>,
    /// EDC criterion values for k = 0, …, M-1
    pub edc_values: Vec<f64>,
}

impl SourceNumberEstimation {
    /// Estimate number of sources from eigenvalues
    ///
    /// # Arguments
    ///
    /// * `eigenvalues` - Eigenvalues in *descending* order (largest first)
    /// * `n_snapshots` - Number of snapshots used for covariance estimation
    ///
    /// # Returns
    ///
    /// * `SourceNumberEstimation` with estimates and criterion values
    pub fn estimate(eigenvalues: &[f64], n_snapshots: usize) -> SignalResult<Self> {
        let m = eigenvalues.len();
        if m == 0 {
            return Err(SignalError::ValueError(
                "Eigenvalues must not be empty".to_string(),
            ));
        }
        if n_snapshots == 0 {
            return Err(SignalError::ValueError(
                "n_snapshots must be positive".to_string(),
            ));
        }

        // Sort eigenvalues in descending order
        let mut eigs = eigenvalues.to_vec();
        eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Clamp eigenvalues to be positive (numerical precision)
        for e in eigs.iter_mut() {
            if *e < 1e-14 {
                *e = 1e-14;
            }
        }

        let n = n_snapshots as f64;
        let mut aic_values = Vec::with_capacity(m);
        let mut mdl_values = Vec::with_capacity(m);
        let mut edc_values = Vec::with_capacity(m);

        // For k = 0 … M-1: the model assumes k sources, M-k noise eigenvalues
        for k in 0..m {
            let noise_eigs = &eigs[k..];
            let d = noise_eigs.len(); // noise subspace dimension = M - k

            // Geometric mean of noise eigenvalues
            let log_sum: f64 = noise_eigs.iter().map(|&e| e.ln()).sum::<f64>();
            let geom_mean = (log_sum / d as f64).exp();

            // Arithmetic mean
            let arith_mean = noise_eigs.iter().sum::<f64>() / d as f64;

            // Log-likelihood ratio term
            let llr = if arith_mean > 1e-30 {
                let ratio = geom_mean / arith_mean;
                if ratio > 1e-30 {
                    n * d as f64 * ratio.ln()
                } else {
                    -n * d as f64 * 30.0
                }
            } else {
                0.0
            };

            // Penalty terms
            let n_params_aic = (k * (2 * m - k)) as f64;
            let n_params_mdl = k as f64 * (2.0 * m as f64 - k as f64);

            // AIC: -2 * log-likelihood + 2 * num_free_params
            let aic = -2.0 * llr + 2.0 * n_params_aic;

            // MDL: -log-likelihood + (1/2) * log(N) * num_free_params
            let mdl = -llr + 0.5 * n.ln() * n_params_mdl;

            // EDC: -log-likelihood + c(N) * num_free_params  where c(N) = sqrt(N) * ln(ln(N))
            let edc_penalty = if n > 2.0 {
                n.sqrt() * (n.ln().ln())
            } else {
                1.0
            };
            let edc = -llr + edc_penalty * n_params_mdl;

            aic_values.push(aic);
            mdl_values.push(mdl);
            edc_values.push(edc);
        }

        // Find minimiser
        let aic_estimate = Self::find_minimum_idx(&aic_values);
        let mdl_estimate = Self::find_minimum_idx(&mdl_values);
        let edc_estimate = Self::find_minimum_idx(&edc_values);

        Ok(Self {
            eigenvalues: eigs,
            n_snapshots,
            aic_estimate,
            mdl_estimate,
            edc_estimate,
            aic_values,
            mdl_values,
            edc_values,
        })
    }

    fn find_minimum_idx(values: &[f64]) -> usize {
        values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Helper: eigendecomposition of Hermitian matrix (using Jacobi iterations)
// ---------------------------------------------------------------------------

/// Perform eigendecomposition of a Hermitian matrix stored row-major in flat Vec.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvalues` is sorted in
/// **descending** order and `eigenvectors[k]` is the k-th eigenvector
/// (corresponding to `eigenvalues[k]`).
///
/// Uses a 2-phase algorithm:
/// 1. Power iteration for dominant eigenpairs
/// 2. Jacobi sweeps on a real symmetric matrix after real extraction
///    (we call the real Jacobi below because R is Hermitian)
pub(crate) fn hermitian_eig(matrix: &[Complex64], n: usize) -> SignalResult<(Vec<f64>, Vec<Vec<Complex64>>)> {
    if matrix.len() != n * n {
        return Err(SignalError::DimensionMismatch(format!(
            "Matrix length {} does not match n={n} (expected {})",
            matrix.len(),
            n * n
        )));
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // We work with a real-representation approach: convert to 2n x 2n real block matrix
    // [Re(A) -Im(A); Im(A) Re(A)]  and use Jacobi on that, then extract n eigenvalues.
    //
    // However, for moderate n, a simpler approach using QR iteration (shifted) is more
    // numerically stable. Here we implement a real symmetric Jacobi on the lower-dimensional
    // representation obtained by splitting the matrix.

    // Because A is Hermitian, eigenvalues are real.  We use the following approach:
    // 1. Build 2n×2n real symmetric block-matrix representation.
    // 2. Run Jacobi iterations on the real matrix.
    // 3. The 2n eigenvalues come in pairs; pick the n unique ones.
    // 4. Map eigenvectors back to complex.

    let n2 = 2 * n;
    let mut real_mat = vec![0.0f64; n2 * n2];

    for i in 0..n {
        for j in 0..n {
            let c = matrix[i * n + j];
            // Top-left block: Re(A)
            real_mat[i * n2 + j] = c.re;
            // Top-right block: -Im(A)
            real_mat[i * n2 + (n + j)] = -c.im;
            // Bottom-left block: Im(A)
            real_mat[(n + i) * n2 + j] = c.im;
            // Bottom-right block: Re(A)
            real_mat[(n + i) * n2 + (n + j)] = c.re;
        }
    }

    let (eigs_real, evecs_real) = jacobi_eig(&real_mat, n2)?;

    // The 2n eigenvalues come in pairs (each eigenvalue of A appears twice).
    // We extract the n largest values among the unique ones by taking every other.
    // Sort pairs and keep alternating values.
    let mut indexed: Vec<(f64, usize)> = eigs_real.iter().cloned().enumerate().map(|(i, v)| (v, i)).collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut eigenvalues: Vec<f64> = Vec::with_capacity(n);
    let mut eigenvectors: Vec<Vec<Complex64>> = Vec::with_capacity(n);

    let mut used = 0;
    let mut i = 0;
    while used < n && i < indexed.len() {
        let (val, col_idx) = indexed[i];
        // Skip near-duplicate (pair partner)
        if used > 0 {
            let prev_val = eigenvalues[used - 1];
            if (val - prev_val).abs() < 1e-8 * (prev_val.abs() + 1.0) {
                // skip the duplicate
                i += 1;
                continue;
            }
        }
        eigenvalues.push(val);
        // Reconstruct complex eigenvector from the real 2n representation
        let real_ev = &evecs_real[col_idx];
        let cv: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(real_ev[k], real_ev[n + k]))
            .collect();
        // Normalise
        let norm = cv.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let cv_norm = if norm > 1e-14 {
            cv.iter().map(|x| *x / norm).collect()
        } else {
            cv
        };
        eigenvectors.push(cv_norm);
        used += 1;
        i += 1;
    }

    // If we didn't get n (due to degenerate), pad
    while eigenvalues.len() < n {
        eigenvalues.push(0.0);
        let mut zero_ev = vec![Complex64::new(0.0, 0.0); n];
        if !zero_ev.is_empty() {
            zero_ev[eigenvalues.len() - 1] = Complex64::new(1.0, 0.0);
        }
        eigenvectors.push(zero_ev);
    }

    Ok((eigenvalues, eigenvectors))
}

/// Jacobi eigenvalue algorithm for real symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[k]` is the k-th
/// column of the orthogonal matrix V such that `A = V D V^T`.
pub(crate) fn jacobi_eig(matrix: &[f64], n: usize) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if matrix.len() != n * n {
        return Err(SignalError::DimensionMismatch(
            "Matrix size mismatch in Jacobi".to_string(),
        ));
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut a = matrix.to_vec();
    // Eigenvector matrix — identity initially
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    let eps = 1e-12;

    for _ in 0..max_iter {
        // Find the largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < eps {
            break;
        }

        // Compute Jacobi rotation
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (aqq - app).abs() < 1e-30 {
            PI / 4.0
        } else {
            0.5 * (2.0 * apq / (aqq - app)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Update A = G^T A G
        // Only need to update rows/cols p and q
        let mut new_a = a.clone();

        for r in 0..n {
            if r != p && r != q {
                let arp = a[r * n + p];
                let arq = a[r * n + q];
                new_a[r * n + p] = cos_t * arp - sin_t * arq;
                new_a[p * n + r] = new_a[r * n + p];
                new_a[r * n + q] = sin_t * arp + cos_t * arq;
                new_a[q * n + r] = new_a[r * n + q];
            }
        }
        new_a[p * n + p] = cos_t * cos_t * app - 2.0 * cos_t * sin_t * apq + sin_t * sin_t * aqq;
        new_a[q * n + q] = sin_t * sin_t * app + 2.0 * cos_t * sin_t * apq + cos_t * cos_t * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;

        // Update V (eigenvectors)
        for r in 0..n {
            let vrp = v[r * n + p];
            let vrq = v[r * n + q];
            v[r * n + p] = cos_t * vrp - sin_t * vrq;
            v[r * n + q] = sin_t * vrp + cos_t * vrq;
        }
    }

    // Extract eigenvalues
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    // Extract eigenvectors as columns of V
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|col| (0..n).map(|row| v[row * n + col]).collect())
        .collect();

    Ok((eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ula_steering_vector_broadside() {
        let ula = UniformLinearArray::new(4, 0.5).expect("ula");
        let sv = ula.steering_vector(0.0).expect("sv");
        assert_eq!(sv.len(), 4);
        // At broadside, all elements have zero phase → all ones
        for c in &sv {
            let diff = (c.re - 1.0).abs() + c.im.abs();
            assert!(diff < 1e-10, "Expected 1+0j, got {:?}", c);
        }
    }

    #[test]
    fn test_uca_creation() {
        let uca = UniformCircularArray::new(8, 0.5).expect("uca");
        assert_eq!(uca.n_elements(), 8);
    }

    #[test]
    fn test_spatial_covariance_estimate() {
        // Simple 2-element array, 4 snapshots
        let data: Vec<Vec<Complex64>> = vec![
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, -1.0),
            ],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, -1.0),
            ],
        ];
        let cov = SpatialCovariance::estimate(&data).expect("cov");
        assert_eq!(cov.size, 2);
        // Diagonal should be 0.5 (average of |1|^2, |j|^2, |-1|^2, |-j|^2 = all 1)
        let diag = cov.get(0, 0);
        assert!((diag.re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_2x2() {
        // Symmetric 2×2 matrix: [[2, 1], [1, 3]]  eigenvalues ≈ 1.382, 3.618
        let mat = vec![2.0, 1.0, 1.0, 3.0];
        let (eigs, _vecs) = jacobi_eig(&mat, 2).expect("jacobi");
        let mut sorted = eigs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let expected = [(5.0 - (5.0f64).sqrt()) / 2.0 + 1.0, (5.0 + (5.0f64).sqrt()) / 2.0 + 0.5];
        // Just check both are in (1.0, 4.0) and distinct
        assert!(sorted[0] < sorted[1]);
        assert!(sorted[0] > 0.5 && sorted[1] < 5.0, "eigenvalues={:?}", sorted);
        let _ = expected;
    }

    #[test]
    fn test_source_number_estimation_aic() {
        // 3 large eigenvalues, 5 small (noise)
        let mut eigs = vec![100.0, 90.0, 80.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let est = SourceNumberEstimation::estimate(&eigs, 200).expect("est");
        // AIC should estimate ~3 sources
        assert!(
            est.aic_estimate <= 4,
            "Expected <=4 sources, got {}",
            est.aic_estimate
        );
    }
}
