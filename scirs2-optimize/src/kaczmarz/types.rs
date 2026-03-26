//! Types for Kaczmarz and sketched projection methods
//!
//! Provides configuration structs and result types for Kaczmarz iterations,
//! sketch-based solvers, and sparse sketching transforms.

/// Variant of the Kaczmarz algorithm (extended set).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum KaczmarzVariantExt {
    /// Classical cyclic Kaczmarz: iterate rows in order 0, 1, ..., m-1, 0, ...
    Classical,
    /// Randomized Kaczmarz (Strohmer-Vershynin): sample row i with probability
    /// proportional to ||a_i||^2 / ||A||_F^2.
    Randomized,
    /// Block Kaczmarz: project onto `block_size` rows simultaneously by solving
    /// a small least-squares sub-problem.
    Block,
    /// Greedy Kaczmarz: select the row with the largest absolute residual
    /// |b_i - a_i^T x|.
    Greedy,
    /// Randomized Extended Kaczmarz (REK): alternates a column-space projection
    /// (to handle inconsistency) with a standard row projection. Converges to
    /// the least-squares solution even for inconsistent systems.
    REK,
}

impl Default for KaczmarzVariantExt {
    fn default() -> Self {
        Self::Randomized
    }
}

/// Configuration for the extended Kaczmarz solver.
#[derive(Debug, Clone)]
pub struct KaczmarzConfigExt {
    /// Maximum number of row-update iterations.
    pub max_iter: usize,
    /// Convergence tolerance on residual norm ||Ax - b||.
    pub tol: f64,
    /// Algorithm variant.
    pub variant: KaczmarzVariantExt,
    /// Block size for the Block variant (ignored for other variants).
    pub block_size: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Relaxation parameter omega in (0, 2). omega = 1.0 gives standard update.
    pub relaxation: f64,
}

impl Default for KaczmarzConfigExt {
    fn default() -> Self {
        Self {
            max_iter: 10_000,
            tol: 1e-6,
            variant: KaczmarzVariantExt::Randomized,
            block_size: 32,
            seed: 42,
            relaxation: 1.0,
        }
    }
}

/// Type of sketch matrix.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum SketchTypeExt {
    /// Dense Gaussian sketch: S_{ij} ~ N(0, 1/sqrt(s)).
    Gaussian,
    /// Subsampled Randomized Hadamard Transform: S = sqrt(n/s) R H D.
    SRHT,
    /// Count sketch: each column mapped to one row with random sign.
    CountSketch,
    /// OSNAP (Oblivious Subspace Embedding): block-diagonal sparse sketch.
    OSNAP,
    /// Sparse Johnson-Lindenstrauss transform.
    SparseJL,
}

impl Default for SketchTypeExt {
    fn default() -> Self {
        Self::Gaussian
    }
}

/// Configuration for sketch-based solvers.
#[derive(Debug, Clone)]
pub struct SketchConfig {
    /// Type of sketch matrix to generate.
    pub sketch_type: SketchTypeExt,
    /// Number of rows in the sketch (sketch dimension s).
    pub sketch_size: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Maximum iterations for iterative sketching.
    pub max_iter: usize,
    /// Convergence tolerance for iterative methods.
    pub tol: f64,
    /// Sparsity parameter for SparseJL (entries per column). If 0, uses O(log n).
    pub sparse_jl_sparsity: usize,
    /// Block count for OSNAP. If 0, chosen automatically.
    pub osnap_blocks: usize,
}

impl Default for SketchConfig {
    fn default() -> Self {
        Self {
            sketch_type: SketchTypeExt::Gaussian,
            sketch_size: 128,
            seed: 42,
            max_iter: 100,
            tol: 1e-6,
            sparse_jl_sparsity: 0,
            osnap_blocks: 0,
        }
    }
}

/// Result of a projection-based or sketch-based solver.
#[derive(Debug, Clone)]
pub struct ProjectionResult {
    /// Approximate solution vector.
    pub solution: Vec<f64>,
    /// Euclidean norm of the final residual ||Ax - b||.
    pub residual_norm: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within the tolerance.
    pub converged: bool,
}
