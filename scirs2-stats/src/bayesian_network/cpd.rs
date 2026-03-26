//! Conditional Probability Distributions (CPDs) for Bayesian Networks.
//!
//! Provides:
//! - [`CPD`] trait — common interface for all CPDs
//! - [`TabularCPD`] — discrete CPD stored as a table
//! - [`GaussianCPD`] — linear Gaussian CPD
//! - [`MixtureCPD`] — mixture of TabularCPDs
//! - [`ConditionalLinear`] — conditional linear Gaussian for continuous parents

use crate::StatsError;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for conditional probability distributions.
pub trait CPD: Send + Sync {
    /// Name of the node this CPD belongs to.
    fn node(&self) -> usize;

    /// Evaluate the (unnormalised) probability (or density) of `value` given
    /// `parent_values`.
    ///
    /// For discrete CPDs this returns the exact probability P(X=value | pa(X) = parent_values).
    /// For continuous CPDs this returns the probability density.
    fn prob(&self, value: usize, parent_values: &[usize]) -> f64;

    /// Number of states (cardinality) for discrete nodes; 0 for continuous.
    fn cardinality(&self) -> usize;

    /// Parent node indices.
    fn parent_indices(&self) -> &[usize];

    /// Whether this CPD is continuous.
    fn is_continuous(&self) -> bool {
        false
    }

    /// Log-probability: default implementation wraps `prob`.
    fn log_prob(&self, value: usize, parent_values: &[usize]) -> f64 {
        let p = self.prob(value, parent_values);
        if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            p.ln()
        }
    }
}

// ---------------------------------------------------------------------------
// TabularCPD
// ---------------------------------------------------------------------------

/// Discrete CPD stored as a conditional probability table (CPT).
///
/// The table is indexed by the combined parent configuration.
/// `values[row][val]` = P(X = val | parent_config = row).
///
/// Row indexing follows *column-major* (rightmost parent cycles fastest),
/// matching pgmpy convention:
///   `row = Sum_i (parent_values[i] * stride[i])`
/// where `stride[i]` = product of cardinalities of parents to the right.
#[derive(Debug, Clone)]
pub struct TabularCPD {
    /// Index of the node this CPD belongs to.
    pub node_idx: usize,
    /// Number of values (cardinality) of this node.
    pub n_values: usize,
    /// Cardinalities of parent nodes.
    pub parent_card: Vec<usize>,
    /// Parent node indices.
    pub parent_indices: Vec<usize>,
    /// Table: `table[row]` = probability distribution over `n_values` states.
    /// Length = product(parent_card), each inner vec has length n_values.
    pub table: Vec<Vec<f64>>,
    /// Strides for row computation.
    strides: Vec<usize>,
}

impl TabularCPD {
    /// Create a new TabularCPD.
    ///
    /// # Arguments
    /// - `node_idx`: Index of the node.
    /// - `n_values`: Cardinality of this node.
    /// - `parent_indices`: Indices of parent nodes.
    /// - `parent_card`: Cardinalities of each parent (same order as parent_indices).
    /// - `values`: Probability table. Each row is a probability distribution.
    ///   If there are no parents, `values` should have exactly one row.
    pub fn new(
        node_idx: usize,
        n_values: usize,
        parent_indices: Vec<usize>,
        parent_card: Vec<usize>,
        values: Vec<Vec<f64>>,
    ) -> Result<Self, StatsError> {
        if parent_indices.len() != parent_card.len() {
            return Err(StatsError::InvalidInput(
                "parent_indices and parent_card must have the same length".to_string(),
            ));
        }
        let n_rows: usize = if parent_card.is_empty() {
            1
        } else {
            parent_card.iter().product()
        };
        if values.len() != n_rows {
            return Err(StatsError::InvalidInput(format!(
                "Expected {n_rows} rows (product of parent cardinalities), got {}",
                values.len()
            )));
        }
        for (i, row) in values.iter().enumerate() {
            if row.len() != n_values {
                return Err(StatsError::InvalidInput(format!(
                    "Row {i} has {} values, expected {n_values}",
                    row.len()
                )));
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(StatsError::InvalidInput(format!(
                    "Row {i} does not sum to 1.0 (sum={sum:.6})"
                )));
            }
        }
        // Compute strides: stride[i] = product(parent_card[i+1..])
        let strides = compute_strides(&parent_card);
        Ok(Self {
            node_idx,
            n_values,
            parent_card,
            parent_indices,
            table: values,
            strides,
        })
    }

    /// Compute the row index for a given parent configuration.
    pub fn row_index(&self, parent_values: &[usize]) -> Result<usize, StatsError> {
        if parent_values.len() != self.parent_card.len() {
            return Err(StatsError::InvalidInput(format!(
                "Expected {} parent values, got {}",
                self.parent_card.len(),
                parent_values.len()
            )));
        }
        let mut row = 0usize;
        for (i, &pv) in parent_values.iter().enumerate() {
            if pv >= self.parent_card[i] {
                return Err(StatsError::InvalidInput(format!(
                    "Parent {i} value {pv} out of range (card={})",
                    self.parent_card[i]
                )));
            }
            row += pv * self.strides[i];
        }
        Ok(row)
    }

    /// Return the full conditional distribution P(X | parent_values).
    pub fn distribution(&self, parent_values: &[usize]) -> Result<&[f64], StatsError> {
        let row = self.row_index(parent_values)?;
        Ok(&self.table[row])
    }
}

impl CPD for TabularCPD {
    fn node(&self) -> usize {
        self.node_idx
    }

    fn prob(&self, value: usize, parent_values: &[usize]) -> f64 {
        if value >= self.n_values {
            return 0.0;
        }
        let row = match self.row_index(parent_values) {
            Ok(r) => r,
            Err(_) => return 0.0,
        };
        self.table[row][value]
    }

    fn cardinality(&self) -> usize {
        self.n_values
    }

    fn parent_indices(&self) -> &[usize] {
        &self.parent_indices
    }
}

// ---------------------------------------------------------------------------
// GaussianCPD
// ---------------------------------------------------------------------------

/// Linear Gaussian CPD: X | pa(X) ~ N(mu + beta^T * pa(X), sigma^2).
///
/// For a root node (no parents), this is simply N(mu, sigma^2).
#[derive(Debug, Clone)]
pub struct GaussianCPD {
    /// Index of this node.
    pub node_idx: usize,
    /// Intercept (mean when all parents are 0).
    pub mu: f64,
    /// Noise standard deviation.
    pub sigma: f64,
    /// Regression coefficients for each parent.
    pub beta: Vec<f64>,
    /// Parent node indices.
    pub parent_indices: Vec<usize>,
}

impl GaussianCPD {
    /// Create a new GaussianCPD.
    pub fn new(
        node_idx: usize,
        mu: f64,
        sigma: f64,
        beta: Vec<f64>,
        parent_indices: Vec<usize>,
    ) -> Result<Self, StatsError> {
        if sigma <= 0.0 {
            return Err(StatsError::InvalidInput(format!(
                "sigma must be positive, got {sigma}"
            )));
        }
        if beta.len() != parent_indices.len() {
            return Err(StatsError::InvalidInput(
                "beta and parent_indices must have the same length".to_string(),
            ));
        }
        Ok(Self {
            node_idx,
            mu,
            sigma,
            beta,
            parent_indices,
        })
    }

    /// Compute the conditional mean given parent values (as continuous f64).
    pub fn conditional_mean(&self, parent_vals: &[f64]) -> f64 {
        self.mu
            + self
                .beta
                .iter()
                .zip(parent_vals)
                .map(|(b, v)| b * v)
                .sum::<f64>()
    }

    /// Compute the conditional density p(x | pa(X)) given continuous value x.
    pub fn density(&self, x: f64, parent_vals: &[f64]) -> f64 {
        let mean = self.conditional_mean(parent_vals);
        let z = (x - mean) / self.sigma;
        (-0.5 * z * z).exp() / (self.sigma * (2.0 * PI).sqrt())
    }
}

impl CPD for GaussianCPD {
    fn node(&self) -> usize {
        self.node_idx
    }

    /// Returns density evaluated at `value` (cast to f64) with `parent_values` cast to f64.
    fn prob(&self, value: usize, parent_values: &[usize]) -> f64 {
        let pv: Vec<f64> = parent_values.iter().map(|&v| v as f64).collect();
        self.density(value as f64, &pv)
    }

    fn cardinality(&self) -> usize {
        0 // continuous
    }

    fn parent_indices(&self) -> &[usize] {
        &self.parent_indices
    }

    fn is_continuous(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// MixtureCPD
// ---------------------------------------------------------------------------

/// Mixture of TabularCPDs.
///
/// P(X | pa(X)) = Σ_k w_k * P_k(X | pa(X))
#[derive(Debug, Clone)]
pub struct MixtureCPD {
    /// Index of this node.
    pub node_idx: usize,
    /// Component CPDs.
    pub components: Vec<TabularCPD>,
    /// Mixture weights (must sum to 1).
    pub weights: Vec<f64>,
}

impl MixtureCPD {
    /// Create a new MixtureCPD.
    pub fn new(
        node_idx: usize,
        components: Vec<TabularCPD>,
        weights: Vec<f64>,
    ) -> Result<Self, StatsError> {
        if components.is_empty() {
            return Err(StatsError::InvalidInput(
                "MixtureCPD needs at least one component".to_string(),
            ));
        }
        if components.len() != weights.len() {
            return Err(StatsError::InvalidInput(
                "components and weights must have the same length".to_string(),
            ));
        }
        let wsum: f64 = weights.iter().sum();
        if (wsum - 1.0).abs() > 1e-6 {
            return Err(StatsError::InvalidInput(format!(
                "weights must sum to 1.0 (got {wsum:.6})"
            )));
        }
        for w in &weights {
            if *w < 0.0 {
                return Err(StatsError::InvalidInput(
                    "weights must be non-negative".to_string(),
                ));
            }
        }
        Ok(Self {
            node_idx,
            components,
            weights,
        })
    }
}

impl CPD for MixtureCPD {
    fn node(&self) -> usize {
        self.node_idx
    }

    fn prob(&self, value: usize, parent_values: &[usize]) -> f64 {
        self.components
            .iter()
            .zip(&self.weights)
            .map(|(c, w)| w * c.prob(value, parent_values))
            .sum()
    }

    fn cardinality(&self) -> usize {
        self.components[0].cardinality()
    }

    fn parent_indices(&self) -> &[usize] {
        self.components[0].parent_indices()
    }
}

// ---------------------------------------------------------------------------
// ConditionalLinear
// ---------------------------------------------------------------------------

/// Conditional Linear Gaussian for continuous parents and discrete output.
///
/// `P(X=k | pa(X)) = softmax(W[k] * pa(X) + b[k])`
/// `sigma[k]` stores standard deviations (for density evaluation).
#[derive(Debug, Clone)]
pub struct ConditionalLinear {
    /// Index of this node.
    pub node_idx: usize,
    /// Weight matrix: `W[k]` has length = number of parents.
    pub w: Vec<Vec<f64>>,
    /// Bias vector: `b[k]` for each class k.
    pub b: Vec<f64>,
    /// Standard deviations (used if output is also continuous).
    pub sigma: Vec<f64>,
    /// Number of output classes.
    pub n_classes: usize,
    /// Parent node indices.
    pub parent_indices: Vec<usize>,
}

impl ConditionalLinear {
    /// Create a new ConditionalLinear CPD.
    pub fn new(
        node_idx: usize,
        w: Vec<Vec<f64>>,
        b: Vec<f64>,
        sigma: Vec<f64>,
        n_classes: usize,
        parent_indices: Vec<usize>,
    ) -> Result<Self, StatsError> {
        if w.len() != n_classes || b.len() != n_classes || sigma.len() != n_classes {
            return Err(StatsError::InvalidInput(
                "w, b, sigma must all have length n_classes".to_string(),
            ));
        }
        Ok(Self {
            node_idx,
            w,
            b,
            sigma,
            n_classes,
            parent_indices,
        })
    }

    /// Compute softmax probabilities.
    pub fn softmax(&self, parent_values: &[f64]) -> Vec<f64> {
        let logits: Vec<f64> = self
            .w
            .iter()
            .zip(&self.b)
            .map(|(wk, bk)| {
                bk + wk
                    .iter()
                    .zip(parent_values)
                    .map(|(wi, xi)| wi * xi)
                    .sum::<f64>()
            })
            .collect();
        let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }
}

impl CPD for ConditionalLinear {
    fn node(&self) -> usize {
        self.node_idx
    }

    fn prob(&self, value: usize, parent_values: &[usize]) -> f64 {
        if value >= self.n_classes {
            return 0.0;
        }
        let pv: Vec<f64> = parent_values.iter().map(|&v| v as f64).collect();
        let probs = self.softmax(&pv);
        probs[value]
    }

    fn cardinality(&self) -> usize {
        self.n_classes
    }

    fn parent_indices(&self) -> &[usize] {
        &self.parent_indices
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute stride array for column-major ordering.
/// stride[i] = product(card[i+1..])
pub(crate) fn compute_strides(card: &[usize]) -> Vec<usize> {
    let n = card.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * card[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rain_cpd() -> TabularCPD {
        // P(Rain=0) = 0.8, P(Rain=1) = 0.2
        TabularCPD::new(0, 2, vec![], vec![], vec![vec![0.8, 0.2]]).unwrap()
    }

    fn wetgrass_cpd() -> TabularCPD {
        // P(WG | Rain, Sprinkler) — 4 rows
        TabularCPD::new(
            2,
            2,
            vec![0, 1], // Rain, Sprinkler
            vec![2, 2],
            vec![
                vec![0.99, 0.01], // R=0, S=0
                vec![0.01, 0.99], // R=0, S=1
                vec![0.01, 0.99], // R=1, S=0
                vec![0.01, 0.99], // R=1, S=1
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_tabular_no_parents() {
        let cpd = rain_cpd();
        assert!((cpd.prob(0, &[]) - 0.8).abs() < 1e-9);
        assert!((cpd.prob(1, &[]) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_tabular_with_parents() {
        let cpd = wetgrass_cpd();
        // P(WG=1 | Rain=1, Spr=0) = 0.99
        assert!((cpd.prob(1, &[1, 0]) - 0.99).abs() < 1e-9);
        // P(WG=0 | Rain=0, Spr=0) = 0.99
        assert!((cpd.prob(0, &[0, 0]) - 0.99).abs() < 1e-9);
    }

    #[test]
    fn test_tabular_bad_sum() {
        let res = TabularCPD::new(0, 2, vec![], vec![], vec![vec![0.5, 0.3]]);
        assert!(res.is_err());
    }

    #[test]
    fn test_gaussian_cpd() {
        let cpd = GaussianCPD::new(0, 0.0, 1.0, vec![0.5], vec![1]).unwrap();
        // Mean = 0 + 0.5 * 2.0 = 1.0; evaluate density at x=1.0
        let d = cpd.density(1.0, &[2.0]);
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((d - expected).abs() < 1e-9);
    }

    #[test]
    fn test_mixture_cpd() {
        let c1 = TabularCPD::new(0, 2, vec![], vec![], vec![vec![0.6, 0.4]]).unwrap();
        let c2 = TabularCPD::new(0, 2, vec![], vec![], vec![vec![0.4, 0.6]]).unwrap();
        let mix = MixtureCPD::new(0, vec![c1, c2], vec![0.5, 0.5]).unwrap();
        // Expected: 0.5*0.6 + 0.5*0.4 = 0.5
        assert!((mix.prob(0, &[]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_conditional_linear() {
        // Two classes, one parent
        let cpd = ConditionalLinear::new(
            0,
            vec![vec![1.0], vec![-1.0]], // w
            vec![0.0, 0.0],              // b
            vec![1.0, 1.0],              // sigma
            2,
            vec![1],
        )
        .unwrap();
        // parent_val = 0 → logits = [0, 0] → softmax = [0.5, 0.5]
        assert!((cpd.prob(0, &[0]) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_strides() {
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }
}
