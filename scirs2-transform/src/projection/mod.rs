//! Random Projection Methods
//!
//! This module implements random projection methods for fast dimensionality
//! reduction, with theoretical guarantees provided by the Johnson-Lindenstrauss
//! lemma: pairwise distances are approximately preserved when projecting from
//! ℝ^d to ℝ^k with k = O(log n / ε²).
//!
//! ## Methods
//!
//! - [`GaussianRandomProjection`]: Dense projection with entries ~ N(0, 1/k).
//!   Guarantees (1±ε) distortion of all pairwise distances with high probability.
//!
//! - [`SparseRandomProjection`]: Achlioptas sparse projection with density
//!   parameter. Entries are ±1/sqrt(s) with prob. s/2 each, 0 with prob. 1-s.
//!
//! - [`AchliptasProjection`]: Special case of sparse projection with density=1/3
//!   (Achlioptas, 2003): entries ±sqrt(3) with prob. 1/6, 0 with prob. 2/3.
//!
//! - [`VerySparseProjn`]: Li et al. (2006) very sparse random projection with
//!   density = 1/sqrt(d).
//!
//! - [`johnson_lindenstrauss_min_components`]: Compute the minimum number of
//!   components needed for (ε, δ)-JL embedding.
//!
//! ## References
//!
//! - Johnson, W.B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
//!   mappings into a Hilbert space. Contemporary Mathematics.
//! - Achlioptas, D. (2003). Database-friendly random projections:
//!   Johnson-Lindenstrauss with binary coins. JCSS.
//! - Li, P., Hastie, T.J., & Church, K.W. (2006). Very sparse random
//!   projections. KDD.
//! - Dasgupta, S., & Gupta, A. (2003). An elementary proof of a theorem of
//!   Johnson and Lindenstrauss. Random Structures & Algorithms.

use scirs2_core::random::{seeded_rng, Distribution, Normal, SeedableRng, Uniform};

use crate::error::{Result, TransformError};

// ============================================================================
// Utility: compute minimum JL components
// ============================================================================

/// Compute the minimum number of projection components required by the
/// Johnson-Lindenstrauss lemma.
///
/// Given n data points and a distortion parameter ε ∈ (0, 1), returns the
/// smallest k such that the JL embedding guarantees (1 ± ε) preservation of
/// all O(n²) pairwise distances with high probability.
///
/// Formula: k = ceil(4 log(n) / (ε²/2 - ε³/3))
///
/// # Arguments
///
/// * `n_samples` - Number of data points.
/// * `eps` - Distortion parameter ε ∈ (0, 1).
///
/// # Returns
///
/// Minimum number of components k.
pub fn johnson_lindenstrauss_min_components(n_samples: usize, eps: f64) -> Result<usize> {
    if eps <= 0.0 || eps >= 1.0 {
        return Err(TransformError::InvalidInput(format!(
            "eps must be in (0, 1), got {eps}"
        )));
    }
    if n_samples < 2 {
        return Err(TransformError::InvalidInput(
            "n_samples must be >= 2".to_string(),
        ));
    }

    let numerator = 4.0 * (n_samples as f64).ln();
    let denominator = eps * eps / 2.0 - eps * eps * eps / 3.0;

    if denominator <= 0.0 {
        return Err(TransformError::ComputationError(
            "JL formula denominator <= 0 (eps too large)".to_string(),
        ));
    }

    let k = (numerator / denominator).ceil() as usize;
    Ok(k.max(1))
}

// ============================================================================
// Helper: project a batch of samples
// ============================================================================

fn project_samples(x: &[Vec<f64>], matrix: &[Vec<f64>], scale: f64) -> Result<Vec<Vec<f64>>> {
    let n = x.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let d_in = x[0].len();
    let d_out = matrix.len();

    let mut out = vec![vec![0.0f64; d_out]; n];
    for (i, row) in x.iter().enumerate() {
        if row.len() != d_in {
            return Err(TransformError::InvalidInput(format!(
                "Row {i}: expected {d_in} features, got {}",
                row.len()
            )));
        }
        for (j, proj_row) in matrix.iter().enumerate() {
            let dot: f64 = proj_row.iter().zip(row.iter()).map(|(p, xi)| p * xi).sum();
            out[i][j] = dot * scale;
        }
    }
    Ok(out)
}

// ============================================================================
// GaussianRandomProjection
// ============================================================================

/// Gaussian random projection: reduces d-dimensional data to k dimensions.
///
/// The projection matrix R ∈ ℝ^{k × d} has entries R_{ij} ~ N(0, 1).
/// The projected data is X' = X R^T / sqrt(k).
///
/// The Johnson-Lindenstrauss lemma guarantees that with k chosen according to
/// [`johnson_lindenstrauss_min_components`], all pairwise distances are
/// preserved within a factor of (1 ± ε) with high probability.
#[derive(Debug, Clone)]
pub struct GaussianRandomProjection {
    /// Target dimension k.
    pub n_components: usize,
    /// Distortion parameter ε (used only for automatic k selection).
    pub eps: f64,
    /// Projection matrix: shape (n_components × n_features).
    projection_matrix: Option<Vec<Vec<f64>>>,
    /// Input feature dimension (set during fit).
    n_features: Option<usize>,
}

impl GaussianRandomProjection {
    /// Create a projection with a fixed number of components.
    pub fn new(n_components: usize) -> Result<Self> {
        if n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }
        Ok(GaussianRandomProjection {
            n_components,
            eps: 0.1,
            projection_matrix: None,
            n_features: None,
        })
    }

    /// Create a projection with automatic component selection via JL lemma.
    pub fn auto(eps: f64, n_samples: usize) -> Result<Self> {
        let k = johnson_lindenstrauss_min_components(n_samples, eps)?;
        Ok(GaussianRandomProjection {
            n_components: k,
            eps,
            projection_matrix: None,
            n_features: None,
        })
    }

    /// Fit the projection matrix to the input dimension.
    pub fn fit(&mut self, n_features: usize, seed: u64) -> Result<()> {
        if n_features == 0 {
            return Err(TransformError::InvalidInput(
                "n_features must be > 0".to_string(),
            ));
        }

        let dist = Normal::new(0.0_f64, 1.0).map_err(|e| {
            TransformError::ComputationError(format!("Normal dist: {e}"))
        })?;
        let mut rng = seeded_rng(seed);

        let matrix: Vec<Vec<f64>> = (0..self.n_components)
            .map(|_| (0..n_features).map(|_| dist.sample(&mut rng)).collect())
            .collect();

        self.projection_matrix = Some(matrix);
        self.n_features = Some(n_features);
        Ok(())
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, x: &[Vec<f64>], seed: u64) -> Result<Vec<Vec<f64>>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        let n_features = x[0].len();
        self.fit(n_features, seed)?;
        self.transform(x)
    }

    /// Project data using the fitted projection matrix.
    /// Scale = 1/sqrt(n_components) for distance preservation.
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let matrix = self.projection_matrix.as_ref().ok_or_else(|| {
            TransformError::NotFitted("GaussianRandomProjection not fitted".to_string())
        })?;
        let scale = 1.0 / (self.n_components as f64).sqrt();
        project_samples(x, matrix, scale)
    }

    /// Return the projection matrix (n_components × n_features).
    pub fn components(&self) -> Option<&Vec<Vec<f64>>> {
        self.projection_matrix.as_ref()
    }
}

// ============================================================================
// SparseRandomProjection
// ============================================================================

/// Sparse random projection (Achlioptas, 2003).
///
/// Projection matrix entries:
/// - +sqrt(1/density) with probability density/2
/// - 0               with probability 1 - density
/// - -sqrt(1/density) with probability density/2
///
/// For density = 1 this recovers Gaussian-like projections (with Rademacher distribution).
/// For density = 1/3 this gives the Achlioptas database-friendly projection.
/// For density = 1/sqrt(d) this is the very sparse projection of Li et al. (2006).
#[derive(Debug, Clone)]
pub struct SparseRandomProjection {
    /// Target dimension k.
    pub n_components: usize,
    /// Density of non-zero entries (0 < density <= 1).
    pub density: f64,
    /// Projection matrix: shape (n_components × n_features).
    projection_matrix: Option<Vec<Vec<f64>>>,
    /// Input feature dimension.
    n_features: Option<usize>,
}

impl SparseRandomProjection {
    /// Create with fixed components and density.
    pub fn new(n_components: usize, density: f64) -> Result<Self> {
        if n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }
        if density <= 0.0 || density > 1.0 {
            return Err(TransformError::InvalidInput(format!(
                "density must be in (0, 1], got {density}"
            )));
        }
        Ok(SparseRandomProjection {
            n_components,
            density,
            projection_matrix: None,
            n_features: None,
        })
    }

    /// Fit the projection matrix to the given input dimension.
    pub fn fit(&mut self, n_features: usize, seed: u64) -> Result<()> {
        if n_features == 0 {
            return Err(TransformError::InvalidInput(
                "n_features must be > 0".to_string(),
            ));
        }

        let scale = (1.0 / self.density).sqrt();
        let u_dist = Uniform::new(0.0_f64, 1.0_f64).map_err(|e| {
            TransformError::ComputationError(format!("Uniform dist: {e}"))
        })?;
        let mut rng = seeded_rng(seed);

        let half_density = self.density / 2.0;
        let matrix: Vec<Vec<f64>> = (0..self.n_components)
            .map(|_| {
                (0..n_features)
                    .map(|_| {
                        let u: f64 = u_dist.sample(&mut rng);
                        if u < half_density {
                            scale
                        } else if u < self.density {
                            -scale
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        self.projection_matrix = Some(matrix);
        self.n_features = Some(n_features);
        Ok(())
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, x: &[Vec<f64>], seed: u64) -> Result<Vec<Vec<f64>>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        let n_features = x[0].len();
        self.fit(n_features, seed)?;
        self.transform(x)
    }

    /// Project data. Scale = 1/sqrt(n_components) for distance preservation.
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let matrix = self.projection_matrix.as_ref().ok_or_else(|| {
            TransformError::NotFitted("SparseRandomProjection not fitted".to_string())
        })?;
        let scale = 1.0 / (self.n_components as f64).sqrt();
        project_samples(x, matrix, scale)
    }

    /// Fraction of non-zero entries in the fitted projection matrix.
    pub fn sparsity(&self) -> f64 {
        match &self.projection_matrix {
            None => 0.0,
            Some(mat) => {
                let total = mat.len() * mat[0].len();
                if total == 0 {
                    return 0.0;
                }
                let nnz: usize = mat.iter().flat_map(|row| row.iter()).filter(|&&v| v != 0.0).count();
                nnz as f64 / total as f64
            }
        }
    }
}

// ============================================================================
// AchliptasProjection — density = 1/3
// ============================================================================

/// Achlioptas (2003) database-friendly random projection with density = 1/3.
///
/// Entries:
/// - +sqrt(3) with probability 1/6
/// - 0        with probability 2/3
/// - -sqrt(3) with probability 1/6
///
/// This requires only 1/3 of the entries to be non-zero, making it fast
/// to apply while maintaining JL distance-preservation guarantees.
#[derive(Debug, Clone)]
pub struct AchliptasProjection {
    /// Target dimension.
    pub n_components: usize,
    /// Inner sparse projection (density = 1/3).
    inner: SparseRandomProjection,
}

impl AchliptasProjection {
    /// Create a new Achlioptas projection.
    pub fn new(n_components: usize) -> Result<Self> {
        let inner = SparseRandomProjection::new(n_components, 1.0 / 3.0)?;
        Ok(AchliptasProjection { n_components, inner })
    }

    /// Fit to the given input dimension.
    pub fn fit(&mut self, n_features: usize, seed: u64) -> Result<()> {
        self.inner.fit(n_features, seed)
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, x: &[Vec<f64>], seed: u64) -> Result<Vec<Vec<f64>>> {
        self.inner.fit_transform(x, seed)
    }

    /// Project data.
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        self.inner.transform(x)
    }

    /// Fraction of non-zero entries (should be ≈ 1/3).
    pub fn sparsity(&self) -> f64 {
        self.inner.sparsity()
    }
}

// ============================================================================
// VerySparseProjn — density = 1/sqrt(n_features)
// ============================================================================

/// Very sparse random projection (Li, Hastie, Church 2006).
///
/// Uses density = 1 / sqrt(n_features), giving an extremely sparse matrix
/// while still satisfying JL guarantees. Entries:
/// - +sqrt(n_features) with probability 1 / (2 sqrt(n_features))
/// - 0                 with probability 1 - 1/sqrt(n_features)
/// - -sqrt(n_features) with probability 1 / (2 sqrt(n_features))
#[derive(Debug, Clone)]
pub struct VerySparseProjn {
    /// Target dimension.
    pub n_components: usize,
    /// Inner sparse projection (density = 1/sqrt(d), determined at fit time).
    inner: Option<SparseRandomProjection>,
}

impl VerySparseProjn {
    /// Create a new very sparse projection.
    pub fn new(n_components: usize) -> Result<Self> {
        if n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be > 0".to_string(),
            ));
        }
        Ok(VerySparseProjn { n_components, inner: None })
    }

    /// Fit: density = 1/sqrt(n_features).
    pub fn fit(&mut self, n_features: usize, seed: u64) -> Result<()> {
        if n_features == 0 {
            return Err(TransformError::InvalidInput(
                "n_features must be > 0".to_string(),
            ));
        }
        let density = 1.0 / (n_features as f64).sqrt();
        let density = density.min(1.0).max(1e-6);
        let mut proj = SparseRandomProjection::new(self.n_components, density)?;
        proj.fit(n_features, seed)?;
        self.inner = Some(proj);
        Ok(())
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, x: &[Vec<f64>], seed: u64) -> Result<Vec<Vec<f64>>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        let n_features = x[0].len();
        self.fit(n_features, seed)?;
        self.transform(x)
    }

    /// Project data.
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            TransformError::NotFitted("VerySparseProjn not fitted".to_string())
        })?;
        inner.transform(x)
    }

    /// Density used (1/sqrt(n_features)).
    pub fn density(&self) -> Option<f64> {
        self.inner.as_ref().map(|p| p.density)
    }

    /// Fraction of non-zero entries.
    pub fn sparsity(&self) -> f64 {
        self.inner.as_ref().map_or(0.0, |p| p.sparsity())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let dist = Normal::new(0.0_f64, 1.0).expect("Normal");
        let mut rng = seeded_rng(seed);
        (0..n)
            .map(|_| (0..d).map(|_| dist.sample(&mut rng)).collect())
            .collect()
    }

    fn pairwise_sq_dists(x: &[Vec<f64>]) -> Vec<f64> {
        let n = x.len();
        let mut dists = vec![];
        for i in 0..n {
            for j in (i + 1)..n {
                let sq: f64 = x[i].iter().zip(x[j].iter()).map(|(a, b)| (a - b).powi(2)).sum();
                dists.push(sq);
            }
        }
        dists
    }

    #[test]
    fn test_jl_min_components() {
        let k = johnson_lindenstrauss_min_components(1000, 0.1).expect("jl");
        assert!(k > 0);
        // Should scale as log(n) / eps^2
        let k_large_eps = johnson_lindenstrauss_min_components(1000, 0.5).expect("jl");
        assert!(k < johnson_lindenstrauss_min_components(1000, 0.05).expect("jl"));
        assert!(k > k_large_eps);
    }

    #[test]
    fn test_jl_invalid_eps() {
        assert!(johnson_lindenstrauss_min_components(100, 0.0).is_err());
        assert!(johnson_lindenstrauss_min_components(100, 1.0).is_err());
        assert!(johnson_lindenstrauss_min_components(100, -0.1).is_err());
    }

    #[test]
    fn test_gaussian_projection_shape() {
        let x = make_data(50, 20, 0);
        let mut proj = GaussianRandomProjection::new(10).expect("new");
        let z = proj.fit_transform(&x, 42).expect("fit_transform");
        assert_eq!(z.len(), 50);
        assert_eq!(z[0].len(), 10);
    }

    #[test]
    fn test_gaussian_projection_distance_preservation() {
        // Weak test: projected pairwise distances should be in a reasonable range
        let x = make_data(30, 100, 7);
        let d_orig = pairwise_sq_dists(&x);

        let mut proj = GaussianRandomProjection::new(500).expect("new");
        let z = proj.fit_transform(&x, 0).expect("fit_transform");
        let d_proj = pairwise_sq_dists(&z);

        // Ratio of projected to original distances should be close to 1
        let ratios: Vec<f64> = d_orig
            .iter()
            .zip(d_proj.iter())
            .filter(|(&do_, _)| do_ > 1e-6)
            .map(|(&do_, &dp)| dp / do_)
            .collect();

        let mean_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
        assert!(
            mean_ratio > 0.5 && mean_ratio < 2.0,
            "Mean distance ratio {mean_ratio:.3} out of expected range"
        );
    }

    #[test]
    fn test_sparse_projection_shape() {
        let x = make_data(40, 15, 0);
        let mut proj = SparseRandomProjection::new(8, 0.5).expect("new");
        let z = proj.fit_transform(&x, 1).expect("fit_transform");
        assert_eq!(z.len(), 40);
        assert_eq!(z[0].len(), 8);
    }

    #[test]
    fn test_sparse_projection_sparsity() {
        let mut proj = SparseRandomProjection::new(100, 1.0 / 3.0).expect("new");
        proj.fit(50, 0).expect("fit");
        let sp = proj.sparsity();
        // Should be near density = 1/3
        assert!(sp > 0.1 && sp < 0.6, "Sparsity {sp:.3} not near 1/3");
    }

    #[test]
    fn test_achlioptas_projection() {
        let x = make_data(30, 20, 0);
        let mut proj = AchliptasProjection::new(10).expect("new");
        let z = proj.fit_transform(&x, 42).expect("fit_transform");
        assert_eq!(z.len(), 30);
        assert_eq!(z[0].len(), 10);
        // Sparsity should be near 1/3
        let sp = proj.sparsity();
        assert!(sp > 0.05 && sp < 0.7, "Achlioptas sparsity {sp:.3}");
    }

    #[test]
    fn test_very_sparse_projection() {
        let x = make_data(25, 100, 0);
        let mut proj = VerySparseProjn::new(20).expect("new");
        let z = proj.fit_transform(&x, 0).expect("fit_transform");
        assert_eq!(z.len(), 25);
        assert_eq!(z[0].len(), 20);

        // Density should be 1/sqrt(100) = 0.1
        let d = proj.density().expect("density");
        assert!((d - 0.1).abs() < 1e-6, "Density {d:.4} != 0.1");
    }

    #[test]
    fn test_projection_not_fitted() {
        let proj = GaussianRandomProjection::new(5).expect("new");
        assert!(proj.transform(&[vec![1.0, 2.0]]).is_err());

        let proj2 = SparseRandomProjection::new(5, 0.3).expect("new");
        assert!(proj2.transform(&[vec![1.0]]).is_err());

        let proj3 = VerySparseProjn::new(5).expect("new");
        assert!(proj3.transform(&[vec![1.0]]).is_err());
    }
}
