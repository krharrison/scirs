//! Types for ML-guided preconditioner selection.

/// The type of preconditioner to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PreconditionerType {
    /// Jacobi (diagonal) preconditioner.
    Jacobi,
    /// Symmetric Successive Over-Relaxation.
    SSOR,
    /// Incomplete LU with zero fill-in.
    ILU0,
    /// Incomplete Cholesky with zero fill-in.
    IC0,
    /// Algebraic Multigrid.
    AMG,
    /// Sparse Approximate Inverse.
    SPAI,
    /// Polynomial preconditioner.
    Polynomial,
    /// No preconditioner (identity / direct solve).
    None,
}

impl std::fmt::Display for PreconditionerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Jacobi => write!(f, "Jacobi"),
            Self::SSOR => write!(f, "SSOR"),
            Self::ILU0 => write!(f, "ILU(0)"),
            Self::IC0 => write!(f, "IC(0)"),
            Self::AMG => write!(f, "AMG"),
            Self::SPAI => write!(f, "SPAI"),
            Self::Polynomial => write!(f, "Polynomial"),
            Self::None => write!(f, "None"),
            #[allow(unreachable_patterns)]
            _ => write!(f, "Unknown"),
        }
    }
}

/// Numerical features extracted from a sparse matrix for classification.
#[derive(Debug, Clone)]
pub struct MatrixFeatures {
    /// Matrix dimension (rows = cols for square matrices).
    pub n: usize,
    /// Number of non-zero entries.
    pub nnz: usize,
    /// Density = nnz / (n * n).
    pub density: f64,
    /// Maximum number of non-zeros in any single row.
    pub max_row_nnz: usize,
    /// Mean number of non-zeros per row.
    pub mean_row_nnz: f64,
    /// Half-bandwidth: max |i - j| over all stored (i, j).
    pub bandwidth: usize,
    /// Bandwidth ratio: bandwidth / n.
    pub bandwidth_ratio: f64,
    /// Cheap condition number estimate (max diag / min diag).
    pub cond_estimate: f64,
    /// Spectral radius estimate via Gershgorin bound.
    pub spectral_radius: f64,
    /// Diagonal dominance ratio: min_i |a_ii| / sum_{j!=i} |a_ij|.
    pub diag_dominance: f64,
    /// Symmetry measure: fraction of (i,j) entries that have a matching (j,i).
    pub symmetry_measure: f64,
    /// Whether all diagonal entries are positive.
    pub has_positive_diagonal: bool,
}

/// Configuration for preconditioner selection.
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Whether to incorporate the cost model when ranking candidates.
    pub use_cost_model: bool,
    /// Maximum number of features used by the classifier.
    pub max_features: usize,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            use_cost_model: true,
            max_features: 12,
        }
    }
}

/// Result of preconditioner selection.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// The recommended preconditioner type.
    pub recommended: PreconditionerType,
    /// Confidence in the recommendation (0.0 – 1.0).
    pub confidence: f64,
    /// All candidate scores, sorted descending by score.
    pub all_scores: Vec<(PreconditionerType, f64)>,
    /// The matrix features that drove the decision.
    pub features: MatrixFeatures,
}

/// Cost estimate for applying a given preconditioner.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Estimated setup cost (e.g. factorization), in FLOPs.
    pub setup_cost: f64,
    /// Estimated cost per Krylov iteration, in FLOPs.
    pub per_iteration_cost: f64,
    /// Estimated number of iterations to convergence.
    pub estimated_iterations: usize,
    /// Total estimated cost: setup + iterations * per_iteration.
    pub total_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preconditioner_type_display() {
        assert_eq!(format!("{}", PreconditionerType::Jacobi), "Jacobi");
        assert_eq!(format!("{}", PreconditionerType::ILU0), "ILU(0)");
        assert_eq!(format!("{}", PreconditionerType::None), "None");
    }

    #[test]
    fn test_selection_config_default() {
        let cfg = SelectionConfig::default();
        assert!(cfg.use_cost_model);
        assert_eq!(cfg.max_features, 12);
    }

    #[test]
    fn test_preconditioner_type_eq() {
        assert_eq!(PreconditionerType::AMG, PreconditionerType::AMG);
        assert_ne!(PreconditionerType::Jacobi, PreconditionerType::SSOR);
    }
}
