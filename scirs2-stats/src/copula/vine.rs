//! Vine (R-vine and C-vine) copula models.
//!
//! Vine copulas decompose a multivariate distribution into a cascade of bivariate
//! copulas via a sequence of trees (the "vine" structure). They are extremely
//! flexible for capturing complex dependence patterns in dimensions d ≥ 3.
//!
//! # Vine Types
//! - **C-vine (Canonical vine)**: one "central" variable connected to all others at each tree level
//! - **D-vine (Drawable vine)**: a path structure, each variable connected to its neighbors
//!
//! # Mathematical Background
//! For a d-variate distribution, a D-vine uses d(d-1)/2 bivariate copulas in a
//! triangular structure. The conditional CDFs (h-functions) are used for sampling
//! via the Rosenblatt transform.
//!
//! # References
//! - Bedford & Cooke (2002). Vines — a new graphical model for dependent random variables.
//! - Aas et al. (2009). Pair-copula constructions of multiple dependence structures.

use super::archimedean::{ClaytonCopula, FrankCopula, GumbelCopula, LcgRng};
use super::elliptical::GaussianCopula;
use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// PairCopula enum
// ---------------------------------------------------------------------------

/// A bivariate copula used as a building block in vine copulas.
#[derive(Debug, Clone, PartialEq)]
pub enum PairCopula {
    /// Independence copula: C(u,v) = u*v
    Independence,
    /// Gaussian copula with correlation ρ
    Gaussian(f64),
    /// Clayton copula with parameter θ > 0
    Clayton(f64),
    /// Gumbel copula with parameter θ ≥ 1
    Gumbel(f64),
    /// Frank copula with parameter θ ≠ 0
    Frank(f64),
}

impl PairCopula {
    /// Evaluate the copula CDF.
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        match self {
            PairCopula::Independence => (u * v).clamp(0.0, 1.0),
            PairCopula::Gaussian(rho) => GaussianCopula::new(*rho)
                .map(|c| c.cdf(u, v))
                .unwrap_or(u * v),
            PairCopula::Clayton(theta) => ClaytonCopula::new(*theta)
                .map(|c| c.cdf(u, v))
                .unwrap_or(u * v),
            PairCopula::Gumbel(theta) => GumbelCopula::new(*theta)
                .map(|c| c.cdf(u, v))
                .unwrap_or(u * v),
            PairCopula::Frank(theta) => FrankCopula::new(*theta)
                .map(|c| c.cdf(u, v))
                .unwrap_or(u * v),
        }
    }

    /// Evaluate the copula PDF.
    pub fn pdf(&self, u: f64, v: f64) -> f64 {
        match self {
            PairCopula::Independence => 1.0,
            PairCopula::Gaussian(rho) => GaussianCopula::new(*rho)
                .map(|c| c.pdf(u, v))
                .unwrap_or(1.0),
            PairCopula::Clayton(theta) => ClaytonCopula::new(*theta)
                .map(|c| c.pdf(u, v))
                .unwrap_or(1.0),
            PairCopula::Gumbel(theta) => GumbelCopula::new(*theta)
                .map(|c| c.pdf(u, v))
                .unwrap_or(1.0),
            PairCopula::Frank(theta) => FrankCopula::new(*theta)
                .map(|c| c.pdf(u, v))
                .unwrap_or(1.0),
        }
    }

    /// H-function (conditional CDF): h(v|u) = ∂C(u,v)/∂u
    ///
    /// Used in the Rosenblatt transform for vine sampling.
    pub fn h_function(&self, u: f64, v: f64) -> f64 {
        let eps = 1e-6;
        let u_safe = u.clamp(eps, 1.0 - eps);
        let v_safe = v.clamp(eps, 1.0 - eps);
        let h_step = u_safe * 1e-4 + 1e-6;
        // Numerical derivative ∂C/∂u
        let c_plus = self.cdf((u_safe + h_step).min(1.0 - eps), v_safe);
        let c_minus = self.cdf((u_safe - h_step).max(eps), v_safe);
        ((c_plus - c_minus) / (2.0 * h_step)).clamp(0.0, 1.0)
    }

    /// Inverse H-function: h⁻¹(w|u) such that h(v|u) = w
    pub fn h_inverse(&self, u: f64, w: f64) -> f64 {
        let eps = 1e-8;
        let u_s = u.clamp(eps, 1.0 - eps);
        let w_s = w.clamp(eps, 1.0 - eps);
        // Bisect
        let mut lo = eps;
        let mut hi = 1.0 - eps;
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            if self.h_function(u_s, mid) < w_s {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        ((lo + hi) / 2.0).clamp(eps, 1.0 - eps)
    }

    /// Kendall's τ for this pair copula.
    pub fn kendall_tau(&self) -> f64 {
        match self {
            PairCopula::Independence => 0.0,
            PairCopula::Gaussian(rho) => 2.0 / std::f64::consts::PI * rho.asin(),
            PairCopula::Clayton(theta) => theta / (theta + 2.0),
            PairCopula::Gumbel(theta) => 1.0 - 1.0 / theta,
            PairCopula::Frank(theta) => {
                FrankCopula::new(*theta).map(|c| c.kendall_tau()).unwrap_or(0.0)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Vine tree structure
// ---------------------------------------------------------------------------

/// Triangular array of pair copulas representing a vine tree structure.
///
/// For a d-dimensional vine, `pairs[tree][edge]` gives the pair copula
/// at position (tree+1, edge+1) in the triangular array.
///
/// - `order`: variable ordering (length d)
/// - `pairs`: (d-1) trees, each with decreasing number of edges
#[derive(Debug, Clone)]
pub struct VineTree {
    /// Variable ordering (length d)
    pub order: Vec<usize>,
    /// Triangular array of pair copulas: pairs[i][j] is the copula in tree i+1, edge j+1
    pub pairs: Vec<Vec<PairCopula>>,
}

impl VineTree {
    /// Create a new vine tree.
    ///
    /// # Arguments
    /// - `order`: variable ordering of length d ≥ 2
    /// - `pairs`: triangular array with pairs[i].len() == d-1-i for i in 0..d-1
    ///
    /// # Errors
    /// Returns an error if dimensions are inconsistent.
    pub fn new(order: Vec<usize>, pairs: Vec<Vec<PairCopula>>) -> StatsResult<Self> {
        let d = order.len();
        if d < 2 {
            return Err(StatsError::InvalidArgument(
                "Vine tree requires at least 2 variables".into(),
            ));
        }
        if pairs.len() != d - 1 {
            return Err(StatsError::InvalidArgument(format!(
                "Expected {} trees (d-1), got {}", d - 1, pairs.len()
            )));
        }
        for (i, tree) in pairs.iter().enumerate() {
            let expected = d - 1 - i;
            if tree.len() != expected {
                return Err(StatsError::InvalidArgument(format!(
                    "Tree {} should have {} copulas, got {}", i + 1, expected, tree.len()
                )));
            }
        }
        Ok(Self { order, pairs })
    }

    /// Dimension of the vine model.
    pub fn dim(&self) -> usize {
        self.order.len()
    }
}

// ---------------------------------------------------------------------------
// D-Vine
// ---------------------------------------------------------------------------

/// D-vine (drawable vine) copula for multivariate sampling.
///
/// In a D-vine, the pair copulas form a path structure:
/// - Tree 1: (1,2), (2,3), ..., (d-1, d)
/// - Tree 2: (1,3|2), (2,4|3), ...
/// - ...
///
/// Sampling uses the Rosenblatt transform sequentially.
#[derive(Debug, Clone)]
pub struct DVine {
    /// Vine tree structure
    pub tree: VineTree,
}

impl DVine {
    /// Create a D-vine from a vine tree.
    pub fn new(tree: VineTree) -> Self {
        Self { tree }
    }

    /// Create a D-vine with all Gaussian pair copulas (given rho matrix).
    pub fn gaussian(d: usize, rho: f64) -> StatsResult<Self> {
        if d < 2 {
            return Err(StatsError::InvalidArgument("d must be >= 2".into()));
        }
        let order: Vec<usize> = (0..d).collect();
        let mut pairs = Vec::with_capacity(d - 1);
        for i in 0..(d - 1) {
            let n_edges = d - 1 - i;
            let tree_copulas = vec![PairCopula::Gaussian(rho); n_edges];
            pairs.push(tree_copulas);
        }
        VineTree::new(order, pairs).map(DVine::new)
    }

    /// Sample n observations from the D-vine using the Rosenblatt transform.
    ///
    /// Returns a matrix of shape (n, d) where each row is one observation.
    pub fn sample(&self, n: usize, rng: &mut impl LcgRng) -> Vec<Vec<f64>> {
        let d = self.tree.dim();
        let mut result = Vec::with_capacity(n);

        for _ in 0..n {
            let obs = self.sample_one(rng, d);
            result.push(obs);
        }
        result
    }

    /// Sample a single observation from the D-vine.
    fn sample_one(&self, rng: &mut impl LcgRng, d: usize) -> Vec<f64> {
        // Generate d uniform(0,1) samples
        let w: Vec<f64> = (0..d).map(|_| rng.next_unit()).collect();

        // v[i][j] = h^{-1}(w[i] | conditioning set) at tree j
        // We need to invert the Rosenblatt transform
        let mut v = vec![vec![0.0f64; d]; d];
        for j in 0..d {
            v[0][j] = w[j];
        }

        // For D-vine, the Rosenblatt inversion follows a specific recursion
        // Reference: Aas et al. (2009), Algorithm 2
        let mut u = vec![0.0f64; d];

        // First variable: u[0] = w[0]
        u[0] = w[0];

        // Second variable: use first pair copula
        if d >= 2 {
            u[1] = self.tree.pairs[0][0].h_inverse(u[0], w[1]);
        }

        // Subsequent variables
        for i in 2..d {
            u[i] = w[i];
            // Invert through conditioning variables
            for k in (0..i).rev() {
                let tree_idx = k;
                let edge_idx = i - 1 - k;
                if tree_idx < self.tree.pairs.len() && edge_idx < self.tree.pairs[tree_idx].len() {
                    let copula = &self.tree.pairs[tree_idx][edge_idx];
                    let v_cond = v[k][i - k - 1];
                    u[i] = copula.h_inverse(v_cond.clamp(1e-10, 1.0 - 1e-10), u[i]);
                }
            }
            // Update v matrix for future conditioning
            v[1][i - 1] = self.tree.pairs[0][i - 1]
                .h_function(u[i - 1], u[i]);
            for k in 1..(i - 1).min(self.tree.pairs.len()) {
                if k < self.tree.pairs.len() && (i - 1 - k) < self.tree.pairs[k].len() {
                    let v_prev = v[k][i - 1 - k];
                    v[k + 1][i - 1 - k] = self.tree.pairs[k][i - 1 - k]
                        .h_function(v_prev.clamp(1e-10, 1.0 - 1e-10), v[k][i - k]);
                }
            }
        }

        u.iter().map(|&x| x.clamp(1e-10, 1.0 - 1e-10)).collect()
    }

    /// Compute the log-density of the D-vine at a given observation.
    pub fn log_pdf(&self, u: &[f64]) -> f64 {
        let d = self.tree.dim();
        if u.len() != d {
            return f64::NEG_INFINITY;
        }
        let mut log_dens = 0.0;
        // Matrix of conditional values
        let mut v = vec![vec![0.0f64; d]; d];
        for (j, &uj) in u.iter().enumerate() {
            v[0][j] = uj;
        }

        // Tree 1 contributions
        for j in 0..(d - 1) {
            let c = &self.tree.pairs[0][j];
            let p = c.pdf(v[0][j], v[0][j + 1]);
            if p > 0.0 {
                log_dens += p.ln();
            } else {
                return f64::NEG_INFINITY;
            }
            // Compute h-functions for next tree
            v[1][j] = c.h_function(v[0][j], v[0][j + 1]);
        }

        // Higher trees
        for i in 1..(d - 1).min(self.tree.pairs.len()) {
            for j in 0..(d - 1 - i).min(self.tree.pairs[i].len()) {
                let c = &self.tree.pairs[i][j];
                let p = c.pdf(v[i][j], v[i][j + 1]);
                if p > 0.0 {
                    log_dens += p.ln();
                }
                if i + 1 < d && j < d - 2 - i {
                    v[i + 1][j] = c.h_function(v[i][j], v[i][j + 1]);
                }
            }
        }

        log_dens
    }
}

// ---------------------------------------------------------------------------
// C-Vine
// ---------------------------------------------------------------------------

/// C-vine (canonical vine) copula.
///
/// In a C-vine, at each tree level there is one "central" node connected
/// to all others:
/// - Tree 1: (1,2), (1,3), ..., (1,d) — variable 1 is the root
/// - Tree 2: (2,3|1), (2,4|1), ... — variable 2 is the root
/// - ...
#[derive(Debug, Clone)]
pub struct CVine {
    /// Vine tree structure
    pub tree: VineTree,
}

impl CVine {
    /// Create a C-vine from a vine tree.
    pub fn new(tree: VineTree) -> Self {
        Self { tree }
    }

    /// Create a C-vine with all Gaussian pair copulas.
    pub fn gaussian(d: usize, rho: f64) -> StatsResult<Self> {
        if d < 2 {
            return Err(StatsError::InvalidArgument("d must be >= 2".into()));
        }
        let order: Vec<usize> = (0..d).collect();
        let mut pairs = Vec::with_capacity(d - 1);
        for i in 0..(d - 1) {
            let n_edges = d - 1 - i;
            let tree_copulas = vec![PairCopula::Gaussian(rho); n_edges];
            pairs.push(tree_copulas);
        }
        VineTree::new(order, pairs).map(CVine::new)
    }

    /// Sample n observations from the C-vine using the Rosenblatt transform.
    pub fn sample(&self, n: usize, rng: &mut impl LcgRng) -> Vec<Vec<f64>> {
        let d = self.tree.dim();
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let obs = self.sample_one(rng, d);
            result.push(obs);
        }
        result
    }

    /// Sample a single observation from the C-vine.
    fn sample_one(&self, rng: &mut impl LcgRng, d: usize) -> Vec<f64> {
        // Algorithm 1 from Aas et al. (2009)
        let w: Vec<f64> = (0..d).map(|_| rng.next_unit()).collect();
        let mut u = vec![0.0f64; d];
        // v[i][j]: v_{i+1,j} in the h-function matrix
        let mut v = vec![vec![0.0f64; d]; d];

        u[0] = w[0];

        for i in 1..d {
            u[i] = w[i];
            // Apply h_inverse through each conditioning set
            for k in (0..i).rev() {
                if k < self.tree.pairs.len() {
                    let j = i - k - 1;
                    if j < self.tree.pairs[k].len() {
                        let copula = &self.tree.pairs[k][j];
                        // For C-vine, condition on first k+1 variables
                        let v_cond = if k == 0 {
                            u[0]
                        } else {
                            v[k][i - k - 1].clamp(1e-10, 1.0 - 1e-10)
                        };
                        u[i] = copula.h_inverse(v_cond, u[i]);
                    }
                }
            }
            // Update v for the h-functions
            v[1][i - 1] = if !self.tree.pairs.is_empty() && !self.tree.pairs[0].is_empty() {
                let j = i - 1;
                if j < self.tree.pairs[0].len() {
                    self.tree.pairs[0][j].h_function(u[0], u[i])
                } else {
                    u[i]
                }
            } else {
                u[i]
            };

            for k in 1..i.min(self.tree.pairs.len()) {
                let j = i - k - 1;
                if j < self.tree.pairs[k].len() {
                    let v_k = v[k][i - k - 1].clamp(1e-10, 1.0 - 1e-10);
                    v[k + 1][i - k - 1] = self.tree.pairs[k][j].h_function(v_k, u[i]);
                }
            }
        }

        u.iter().map(|&x| x.clamp(1e-10, 1.0 - 1e-10)).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::copula::archimedean::SimpleLcg;

    #[test]
    fn test_pair_copula_cdf_independence() {
        let c = PairCopula::Independence;
        assert!((c.cdf(0.5, 0.5) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_pair_copula_cdf_gaussian() {
        let c = PairCopula::Gaussian(0.5);
        let val = c.cdf(0.5, 0.5);
        assert!(val > 0.0 && val <= 1.0);
    }

    #[test]
    fn test_pair_copula_pdf_positive() {
        for copula in &[
            PairCopula::Gaussian(0.5),
            PairCopula::Clayton(2.0),
            PairCopula::Gumbel(2.0),
            PairCopula::Frank(3.0),
        ] {
            let p = copula.pdf(0.4, 0.6);
            assert!(p > 0.0, "pdf=0 for {:?}", copula);
        }
    }

    #[test]
    fn test_pair_copula_h_function_range() {
        let c = PairCopula::Gaussian(0.5);
        let h = c.h_function(0.3, 0.7);
        assert!(h >= 0.0 && h <= 1.0, "h={h}");
    }

    #[test]
    fn test_pair_copula_h_inverse_roundtrip() {
        let c = PairCopula::Clayton(2.0);
        let u = 0.4;
        let v = 0.6;
        let h = c.h_function(u, v);
        let v_back = c.h_inverse(u, h);
        assert!((v_back - v).abs() < 1e-4, "v={v}, v_back={v_back}");
    }

    #[test]
    fn test_pair_copula_kendall_tau() {
        let c = PairCopula::Independence;
        assert_eq!(c.kendall_tau(), 0.0);
        let g = PairCopula::Gaussian(0.5);
        assert!(g.kendall_tau().is_finite());
    }

    #[test]
    fn test_vine_tree_invalid_dim() {
        assert!(VineTree::new(vec![0], vec![]).is_err());
    }

    #[test]
    fn test_vine_tree_mismatched_pairs() {
        let order = vec![0, 1, 2];
        // Wrong: should have 2 trees for d=3
        let result = VineTree::new(order, vec![vec![PairCopula::Independence]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dvine_gaussian_creation() {
        let dvine = DVine::gaussian(3, 0.5).unwrap();
        assert_eq!(dvine.tree.dim(), 3);
        assert_eq!(dvine.tree.pairs.len(), 2);
    }

    #[test]
    fn test_dvine_sample_dimensions() {
        let dvine = DVine::gaussian(3, 0.5).unwrap();
        let mut rng = SimpleLcg::new(42);
        let samples = dvine.sample(100, &mut rng);
        assert_eq!(samples.len(), 100);
        for obs in &samples {
            assert_eq!(obs.len(), 3);
            for &x in obs {
                assert!(x > 0.0 && x < 1.0, "x={x} out of range");
            }
        }
    }

    #[test]
    fn test_dvine_log_pdf_finite() {
        let dvine = DVine::gaussian(3, 0.5).unwrap();
        let ll = dvine.log_pdf(&[0.3, 0.5, 0.7]);
        // Should be finite (may be positive or negative)
        assert!(ll.is_finite(), "log_pdf={ll}");
    }

    #[test]
    fn test_cvine_gaussian_creation() {
        let cvine = CVine::gaussian(4, 0.6).unwrap();
        assert_eq!(cvine.tree.dim(), 4);
    }

    #[test]
    fn test_cvine_sample_dimensions() {
        let cvine = CVine::gaussian(3, 0.5).unwrap();
        let mut rng = SimpleLcg::new(99);
        let samples = cvine.sample(50, &mut rng);
        assert_eq!(samples.len(), 50);
        for obs in &samples {
            assert_eq!(obs.len(), 3);
            for &x in obs {
                assert!(x > 0.0 && x < 1.0, "x={x}");
            }
        }
    }

    #[test]
    fn test_cvine_invalid_dim() {
        assert!(CVine::gaussian(1, 0.5).is_err());
    }

    #[test]
    fn test_dvine_with_mixed_copulas() {
        // 3-dimensional D-vine with mixed copulas
        let pairs = vec![
            vec![PairCopula::Clayton(2.0), PairCopula::Gumbel(1.5)], // Tree 1
            vec![PairCopula::Gaussian(0.3)], // Tree 2
        ];
        let tree = VineTree::new(vec![0, 1, 2], pairs).unwrap();
        let dvine = DVine::new(tree);
        let mut rng = SimpleLcg::new(17);
        let samples = dvine.sample(20, &mut rng);
        assert_eq!(samples.len(), 20);
        for obs in &samples {
            assert_eq!(obs.len(), 3);
        }
    }
}
