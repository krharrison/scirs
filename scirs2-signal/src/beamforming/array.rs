//! Array geometry definitions and steering vector computation
//!
//! Provides:
//! - [`UniformLinearArray`]: ULA with elements at d * [0, 1, ..., N-1]
//! - [`UniformCircularArray`]: UCA with elements on a circle of radius R
//! - [`ArbitraryArray`]: user-specified element positions
//! - [`ArrayGeometry`]: trait for steering vector computation
//! - `ArrayManifold`: set of steering vectors for a range of angles
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// ArrayGeometry trait
// ---------------------------------------------------------------------------

/// Trait for array geometry — provides steering vector computation
pub trait ArrayGeometry: Send + Sync {
    /// Compute the steering vector for a given angle (radians, measured from broadside)
    /// at a given frequency.
    ///
    /// # Arguments
    ///
    /// * `angle_rad` - Direction angle in radians (0 = broadside)
    /// * `wavelength` - Signal wavelength (same units as element positions)
    fn steering_vector(&self, angle_rad: f64, wavelength: f64) -> SignalResult<Vec<Complex64>>;

    /// Number of array elements
    fn n_elements(&self) -> usize;

    /// Get element positions as (x, y) coordinates
    fn element_positions(&self) -> Vec<(f64, f64)>;

    /// Compute steering vector using element spacing in wavelengths (convenience)
    ///
    /// Uses wavelength = 1.0 so that positions are interpreted directly in wavelengths.
    fn steering_vector_normalized(&self, angle_rad: f64) -> SignalResult<Vec<Complex64>> {
        self.steering_vector(angle_rad, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Uniform Linear Array (ULA)
// ---------------------------------------------------------------------------

/// Uniform Linear Array (ULA)
///
/// Elements are placed at positions d * [0, 1, ..., N-1] along the x-axis.
/// The steering vector is:
///
/// `a(theta) = [1, exp(-j*2*pi*d*sin(theta)/lambda), ..., exp(-j*2*pi*(N-1)*d*sin(theta)/lambda)]`
#[derive(Debug, Clone)]
pub struct UniformLinearArray {
    /// Number of elements
    n_elements: usize,
    /// Inter-element spacing (in metres or wavelengths)
    element_spacing: f64,
}

impl UniformLinearArray {
    /// Create a new ULA
    ///
    /// # Arguments
    ///
    /// * `n_elements` - Number of sensors (must be >= 2)
    /// * `element_spacing` - Spacing between adjacent elements
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

    /// Get element spacing
    pub fn element_spacing(&self) -> f64 {
        self.element_spacing
    }

    /// Compute steering vectors for multiple angles simultaneously
    pub fn steering_matrix(
        &self,
        angles_rad: &[f64],
        wavelength: f64,
    ) -> SignalResult<Vec<Vec<Complex64>>> {
        angles_rad
            .iter()
            .map(|&theta| self.steering_vector(theta, wavelength))
            .collect()
    }
}

impl ArrayGeometry for UniformLinearArray {
    fn steering_vector(&self, angle_rad: f64, wavelength: f64) -> SignalResult<Vec<Complex64>> {
        if wavelength <= 0.0 {
            return Err(SignalError::ValueError(
                "Wavelength must be positive".to_string(),
            ));
        }
        let phase_increment = -2.0 * PI * self.element_spacing * angle_rad.sin() / wavelength;
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

    fn element_positions(&self) -> Vec<(f64, f64)> {
        (0..self.n_elements)
            .map(|m| (self.element_spacing * m as f64, 0.0))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Uniform Circular Array (UCA)
// ---------------------------------------------------------------------------

/// Uniform Circular Array (UCA)
///
/// N elements are placed uniformly on a circle of radius R.
/// Element m is at angle `2*pi*m/N` from the x-axis.
/// For a source at azimuth `phi`, the m-th element phase is:
///
/// `psi_m = 2*pi*(R/lambda)*cos(phi - 2*pi*m/N)`
#[derive(Debug, Clone)]
pub struct UniformCircularArray {
    /// Number of elements
    n_elements: usize,
    /// Array radius
    radius: f64,
}

impl UniformCircularArray {
    /// Create a new UCA
    ///
    /// # Arguments
    ///
    /// * `n_elements` - Number of sensors (must be >= 3)
    /// * `radius` - Array radius (same units as wavelength)
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

    /// Get array radius
    pub fn radius(&self) -> f64 {
        self.radius
    }
}

impl ArrayGeometry for UniformCircularArray {
    fn steering_vector(&self, angle_rad: f64, wavelength: f64) -> SignalResult<Vec<Complex64>> {
        if wavelength <= 0.0 {
            return Err(SignalError::ValueError(
                "Wavelength must be positive".to_string(),
            ));
        }
        let n = self.n_elements;
        let sv = (0..n)
            .map(|m| {
                let element_angle = 2.0 * PI * m as f64 / n as f64;
                let phase = 2.0 * PI * self.radius * (angle_rad - element_angle).cos() / wavelength;
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();
        Ok(sv)
    }

    fn n_elements(&self) -> usize {
        self.n_elements
    }

    fn element_positions(&self) -> Vec<(f64, f64)> {
        let n = self.n_elements;
        (0..n)
            .map(|m| {
                let angle = 2.0 * PI * m as f64 / n as f64;
                (self.radius * angle.cos(), self.radius * angle.sin())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Arbitrary Array
// ---------------------------------------------------------------------------

/// Arbitrary sensor array with user-specified element positions
///
/// Supports any planar geometry. Positions are given as (x, y) coordinates.
/// The steering vector for a far-field source at angle `theta` from broadside
/// (y-axis) is:
///
/// `a_m(theta) = exp(-j*2*pi*(x_m*sin(theta))/lambda)`
///
/// For a full 2D model use the `steering_vector_2d` method.
#[derive(Debug, Clone)]
pub struct ArbitraryArray {
    /// Element positions as (x, y)
    positions: Vec<(f64, f64)>,
}

impl ArbitraryArray {
    /// Create a new arbitrary array
    ///
    /// # Arguments
    ///
    /// * `positions` - Element positions as (x, y) coordinates (must have >= 2 elements)
    pub fn new(positions: Vec<(f64, f64)>) -> SignalResult<Self> {
        if positions.len() < 2 {
            return Err(SignalError::ValueError(
                "Arbitrary array requires at least 2 elements".to_string(),
            ));
        }
        Ok(Self { positions })
    }

    /// Compute the 2D steering vector for azimuth `phi` and elevation `theta`
    ///
    /// Phase: `psi_m = 2*pi*(x_m*cos(phi)*cos(theta) + y_m*sin(phi)*cos(theta)) / lambda`
    pub fn steering_vector_2d(
        &self,
        azimuth_rad: f64,
        elevation_rad: f64,
        wavelength: f64,
    ) -> SignalResult<Vec<Complex64>> {
        if wavelength <= 0.0 {
            return Err(SignalError::ValueError(
                "Wavelength must be positive".to_string(),
            ));
        }
        let cos_el = elevation_rad.cos();
        let kx = 2.0 * PI * azimuth_rad.cos() * cos_el / wavelength;
        let ky = 2.0 * PI * azimuth_rad.sin() * cos_el / wavelength;

        let sv = self
            .positions
            .iter()
            .map(|&(x, y)| {
                let phase = -(kx * x + ky * y);
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();
        Ok(sv)
    }
}

impl ArrayGeometry for ArbitraryArray {
    fn steering_vector(&self, angle_rad: f64, wavelength: f64) -> SignalResult<Vec<Complex64>> {
        if wavelength <= 0.0 {
            return Err(SignalError::ValueError(
                "Wavelength must be positive".to_string(),
            ));
        }
        // 1D model: project positions onto x-axis
        let sv = self
            .positions
            .iter()
            .map(|&(x, _y)| {
                let phase = -2.0 * PI * x * angle_rad.sin() / wavelength;
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();
        Ok(sv)
    }

    fn n_elements(&self) -> usize {
        self.positions.len()
    }

    fn element_positions(&self) -> Vec<(f64, f64)> {
        self.positions.clone()
    }
}

// ---------------------------------------------------------------------------
// Array Manifold
// ---------------------------------------------------------------------------

/// Array manifold: set of steering vectors for a range of angles
#[derive(Debug, Clone)]
pub struct ArrayManifoldData {
    /// Steering vectors, one per scan angle
    pub steering_vectors: Vec<Vec<Complex64>>,
    /// Corresponding scan angles in radians
    pub scan_angles: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
}

impl ArrayManifoldData {
    /// Compute the array manifold for a given geometry and scan range
    ///
    /// # Arguments
    ///
    /// * `array` - Array geometry
    /// * `scan_angles_rad` - Angles to compute steering vectors for
    /// * `wavelength` - Signal wavelength
    pub fn compute(
        array: &dyn ArrayGeometry,
        scan_angles_rad: &[f64],
        wavelength: f64,
    ) -> SignalResult<Self> {
        if scan_angles_rad.is_empty() {
            return Err(SignalError::ValueError(
                "Scan angles must not be empty".to_string(),
            ));
        }
        let mut steering_vectors = Vec::with_capacity(scan_angles_rad.len());
        for &angle in scan_angles_rad {
            steering_vectors.push(array.steering_vector(angle, wavelength)?);
        }
        Ok(Self {
            steering_vectors,
            scan_angles: scan_angles_rad.to_vec(),
            n_elements: array.n_elements(),
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Create uniformly spaced scan angles in radians
///
/// # Arguments
///
/// * `start_deg` - Start angle in degrees
/// * `end_deg` - End angle in degrees
/// * `n_points` - Number of scan points
pub fn scan_angles_degrees(
    start_deg: f64,
    end_deg: f64,
    n_points: usize,
) -> SignalResult<Vec<f64>> {
    if n_points == 0 {
        return Err(SignalError::ValueError(
            "Number of scan points must be positive".to_string(),
        ));
    }
    if n_points == 1 {
        return Ok(vec![start_deg.to_radians()]);
    }
    let step = (end_deg - start_deg) / (n_points - 1) as f64;
    Ok((0..n_points)
        .map(|i| (start_deg + step * i as f64).to_radians())
        .collect())
}

/// Compute ULA steering vector (standalone function for convenience)
///
/// # Arguments
///
/// * `n_elements` - Number of array elements
/// * `angle_rad` - Steering angle in radians
/// * `element_spacing` - Element spacing in wavelengths
pub fn steering_vector_ula(
    n_elements: usize,
    angle_rad: f64,
    element_spacing: f64,
) -> SignalResult<Vec<Complex64>> {
    if n_elements == 0 {
        return Err(SignalError::ValueError(
            "Number of elements must be positive".to_string(),
        ));
    }
    if element_spacing <= 0.0 {
        return Err(SignalError::ValueError(
            "Element spacing must be positive".to_string(),
        ));
    }
    let phase_increment = -2.0 * PI * element_spacing * angle_rad.sin();
    let sv = (0..n_elements)
        .map(|m| {
            let phase = phase_increment * m as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();
    Ok(sv)
}

/// Compute steering vectors for a set of angles
pub fn steering_vectors_ula(
    n_elements: usize,
    angles_rad: &[f64],
    element_spacing: f64,
) -> SignalResult<Vec<Vec<Complex64>>> {
    if angles_rad.is_empty() {
        return Err(SignalError::ValueError(
            "Angle list must not be empty".to_string(),
        ));
    }
    angles_rad
        .iter()
        .map(|&angle| steering_vector_ula(n_elements, angle, element_spacing))
        .collect()
}

// ---------------------------------------------------------------------------
// Covariance matrix estimation
// ---------------------------------------------------------------------------

/// Estimate the spatial covariance matrix from multi-channel complex data
///
/// R = (1/N) * X * X^H
pub fn estimate_covariance(signals: &[Vec<Complex64>]) -> SignalResult<Vec<Vec<Complex64>>> {
    if signals.is_empty() {
        return Err(SignalError::ValueError(
            "Signal matrix must not be empty".to_string(),
        ));
    }
    let n_elements = signals.len();
    let n_snapshots = signals[0].len();
    if n_snapshots == 0 {
        return Err(SignalError::ValueError(
            "Number of snapshots must be positive".to_string(),
        ));
    }
    for (idx, sig) in signals.iter().enumerate() {
        if sig.len() != n_snapshots {
            return Err(SignalError::DimensionMismatch(format!(
                "Element {} has {} snapshots, expected {}",
                idx,
                sig.len(),
                n_snapshots
            )));
        }
    }

    let mut cov = vec![vec![Complex64::new(0.0, 0.0); n_elements]; n_elements];
    for i in 0..n_elements {
        for j in 0..n_elements {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..n_snapshots {
                sum += signals[i][k] * signals[j][k].conj();
            }
            cov[i][j] = sum / n_snapshots as f64;
        }
    }
    Ok(cov)
}

/// Estimate covariance matrix from real-valued multi-channel signals
pub fn estimate_covariance_real(signals: &[Vec<f64>]) -> SignalResult<Vec<Vec<Complex64>>> {
    let complex_signals: Vec<Vec<Complex64>> = signals
        .iter()
        .map(|ch| ch.iter().map(|&x| Complex64::new(x, 0.0)).collect())
        .collect();
    estimate_covariance(&complex_signals)
}

// ---------------------------------------------------------------------------
// Matrix utilities
// ---------------------------------------------------------------------------

/// Invert a Hermitian positive-definite matrix using Gauss-Jordan elimination
pub(crate) fn invert_hermitian_matrix(
    matrix: &[Vec<Complex64>],
) -> SignalResult<Vec<Vec<Complex64>>> {
    let n = matrix.len();
    if n == 0 {
        return Err(SignalError::ValueError(
            "Matrix must not be empty".to_string(),
        ));
    }

    let mut aug: Vec<Vec<Complex64>> = Vec::with_capacity(n);
    for i in 0..n {
        if matrix[i].len() != n {
            return Err(SignalError::DimensionMismatch(format!(
                "Matrix row {} has length {}, expected {}",
                i,
                matrix[i].len(),
                n
            )));
        }
        let mut row = Vec::with_capacity(2 * n);
        row.extend_from_slice(&matrix[i]);
        for j in 0..n {
            if i == j {
                row.push(Complex64::new(1.0, 0.0));
            } else {
                row.push(Complex64::new(0.0, 0.0));
            }
        }
        aug.push(row);
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].norm();
        for row in (col + 1)..n {
            let val = aug[row][col].norm();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(SignalError::ComputationError(
                "Matrix is singular or near-singular".to_string(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        let pivot_inv = Complex64::new(1.0, 0.0) / pivot;
        for j in 0..(2 * n) {
            aug[col][j] = aug[col][j] * pivot_inv;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] = aug[row][j] - factor * aug[col][j];
            }
        }
    }

    let mut inverse = Vec::with_capacity(n);
    for i in 0..n {
        inverse.push(aug[i][n..(2 * n)].to_vec());
    }
    Ok(inverse)
}

/// Matrix-vector product: R * a
pub(crate) fn mat_vec_mul(matrix: &[Vec<Complex64>], vec: &[Complex64]) -> Vec<Complex64> {
    let m = matrix.len();
    let mut result = vec![Complex64::new(0.0, 0.0); m];
    for i in 0..m {
        for (j, &v) in vec.iter().enumerate() {
            result[i] += matrix[i][j] * v;
        }
    }
    result
}

/// Inner product: a^H * b
pub(crate) fn inner_product_conj(a: &[Complex64], b: &[Complex64]) -> Complex64 {
    a.iter()
        .zip(b.iter())
        .fold(Complex64::new(0.0, 0.0), |acc, (&ai, &bi)| {
            acc + ai.conj() * bi
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ula_steering_vector_broadside() {
        let ula = UniformLinearArray::new(4, 0.5).expect("should create ULA");
        let sv = ula
            .steering_vector(0.0, 1.0)
            .expect("should compute steering vector");
        assert_eq!(sv.len(), 4);
        for s in &sv {
            assert_relative_eq!(s.re, 1.0, epsilon = 1e-12);
            assert_relative_eq!(s.im, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_ula_steering_vector_unit_norm() {
        let ula = UniformLinearArray::new(8, 0.5).expect("should create ULA");
        let sv = ula
            .steering_vector(0.3, 1.0)
            .expect("should compute steering vector");
        for s in &sv {
            assert_relative_eq!(s.norm(), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_ula_element_positions() {
        let ula = UniformLinearArray::new(4, 0.5).expect("should create ULA");
        let pos = ula.element_positions();
        assert_eq!(pos.len(), 4);
        assert_relative_eq!(pos[0].0, 0.0, epsilon = 1e-12);
        assert_relative_eq!(pos[1].0, 0.5, epsilon = 1e-12);
        assert_relative_eq!(pos[3].0, 1.5, epsilon = 1e-12);
    }

    #[test]
    fn test_ula_validation() {
        assert!(UniformLinearArray::new(0, 0.5).is_err());
        assert!(UniformLinearArray::new(1, 0.5).is_err());
        assert!(UniformLinearArray::new(4, 0.0).is_err());
        assert!(UniformLinearArray::new(4, -0.5).is_err());
    }

    #[test]
    fn test_uca_steering_vector() {
        let uca = UniformCircularArray::new(8, 1.0).expect("should create UCA");
        let sv = uca
            .steering_vector(0.0, 1.0)
            .expect("should compute steering vector");
        assert_eq!(sv.len(), 8);
        for s in &sv {
            assert_relative_eq!(s.norm(), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_uca_symmetry() {
        let uca = UniformCircularArray::new(8, 1.0).expect("should create UCA");
        // At angle=0, elements symmetric about x-axis should have same magnitude
        let sv = uca
            .steering_vector(0.0, 1.0)
            .expect("should compute steering vector");
        // Element 1 and element 7 are symmetric: same magnitude
        assert_relative_eq!(sv[1].norm(), sv[7].norm(), epsilon = 1e-10);
        // The phases should be conjugate-related
        assert_relative_eq!((sv[1] * sv[7].conj()).im.abs(), 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_uca_validation() {
        assert!(UniformCircularArray::new(2, 1.0).is_err());
        assert!(UniformCircularArray::new(4, 0.0).is_err());
    }

    #[test]
    fn test_arbitrary_array() {
        // Create a ULA-equivalent using arbitrary array
        let positions = vec![(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0)];
        let arr = ArbitraryArray::new(positions).expect("should create arbitrary array");
        assert_eq!(arr.n_elements(), 4);

        let sv = arr
            .steering_vector(0.0, 1.0)
            .expect("should compute steering vector");
        // Broadside: all phases zero
        for s in &sv {
            assert_relative_eq!(s.re, 1.0, epsilon = 1e-12);
            assert_relative_eq!(s.im, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_arbitrary_array_validation() {
        assert!(ArbitraryArray::new(vec![(0.0, 0.0)]).is_err());
    }

    #[test]
    fn test_array_manifold() {
        let ula = UniformLinearArray::new(4, 0.5).expect("should create ULA");
        let angles = scan_angles_degrees(-90.0, 90.0, 181).expect("should create angles");
        let manifold =
            ArrayManifoldData::compute(&ula, &angles, 1.0).expect("should compute manifold");
        assert_eq!(manifold.steering_vectors.len(), 181);
        assert_eq!(manifold.n_elements, 4);
    }

    #[test]
    fn test_scan_angles_degrees() {
        let angles = scan_angles_degrees(-90.0, 90.0, 181).expect("should create angles");
        assert_eq!(angles.len(), 181);
        assert_relative_eq!(angles[0], -PI / 2.0, epsilon = 1e-10);
        assert_relative_eq!(angles[180], PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scan_angles_single() {
        let angles = scan_angles_degrees(30.0, 30.0, 1).expect("should create single angle");
        assert_eq!(angles.len(), 1);
        assert_relative_eq!(angles[0], 30.0_f64.to_radians(), epsilon = 1e-10);
    }

    #[test]
    fn test_scan_angles_validation() {
        assert!(scan_angles_degrees(-90.0, 90.0, 0).is_err());
    }

    #[test]
    fn test_covariance_hermitian() {
        let signals = vec![
            vec![Complex64::new(1.0, 0.5), Complex64::new(0.3, -0.2)],
            vec![Complex64::new(-0.5, 0.1), Complex64::new(0.8, 0.4)],
        ];
        let cov = estimate_covariance(&signals).expect("should compute covariance");
        for i in 0..cov.len() {
            for j in 0..cov.len() {
                assert_relative_eq!(cov[i][j].re, cov[j][i].re, epsilon = 1e-12);
                assert_relative_eq!(cov[i][j].im, -cov[j][i].im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_covariance_real() {
        let signals = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let cov = estimate_covariance_real(&signals).expect("should compute covariance");
        assert_eq!(cov.len(), 2);
        for row in &cov {
            for &val in row {
                assert!(val.im.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_covariance_validation() {
        assert!(estimate_covariance(&[]).is_err());
        assert!(estimate_covariance(&[vec![]]).is_err());
    }

    #[test]
    fn test_invert_identity() {
        let n = 3;
        let mut identity = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            identity[i][i] = Complex64::new(1.0, 0.0);
        }
        let inv = invert_hermitian_matrix(&identity).expect("should invert");
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_relative_eq!(inv[i][j].re, 1.0, epsilon = 1e-10);
                } else {
                    assert!(inv[i][j].norm() < 1e-10);
                }
            }
        }
    }
}
