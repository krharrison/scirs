//! PC-Stable Algorithm for Time Series Causal Discovery
//!
//! Implements the order-independent PC-stable algorithm adapted for time series
//! with lagged variables. This serves as the condition-selection stage (Stage 1)
//! for PCMCI.
//!
//! ## Algorithm Overview
//!
//! 1. Start with a fully connected lagged graph (all (var, lag) pairs connected to all targets)
//! 2. For increasing conditioning set sizes p = 0, 1, 2, ...
//!    - For each edge X_{t-tau} -- Y_t remaining in the skeleton:
//!      - Test if X_{t-tau} _||_ Y_t | S for all subsets S of size p from adjacencies
//!      - If independent: remove the edge and store the separation set
//! 3. The PC-stable variant collects all removals for a given order p before applying them,
//!    ensuring order independence.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use scirs2_core::ndarray::Array2;
//! use scirs2_series::causality::pc_stable::{PCStable, PCStableConfig};
//! use scirs2_series::causality::ci_tests::ParCorr;
//!
//! let data = Array2::zeros((100, 3));
//! let config = PCStableConfig { tau_max: 2, alpha: 0.05, max_cond_size: None };
//! let pc = PCStable::new(Box::new(ParCorr::new()), config);
//! let result = pc.run(&data).expect("PC-stable failed");
//! ```

use std::collections::{HashMap, HashSet};

use crate::error::TimeSeriesError;
use scirs2_core::ndarray::Array2;

use super::ci_tests::{LaggedVar, TimeSeriesCITest};
use super::CausalityResult;

/// Configuration for the PC-stable algorithm
#[derive(Debug, Clone)]
pub struct PCStableConfig {
    /// Maximum lag to consider
    pub tau_max: usize,
    /// Significance level for conditional independence tests
    pub alpha: f64,
    /// Maximum conditioning set size (None = no limit beyond adjacencies)
    pub max_cond_size: Option<usize>,
}

impl Default for PCStableConfig {
    fn default() -> Self {
        Self {
            tau_max: 2,
            alpha: 0.05,
            max_cond_size: None,
        }
    }
}

/// Result of the PC-stable skeleton discovery
#[derive(Debug, Clone)]
pub struct PCStableResult {
    /// For each target variable, the set of parents found: target_var -> Vec<(source_var, lag)>
    pub parents: HashMap<usize, Vec<LaggedVar>>,
    /// Separation sets: (x, y) -> conditioning set that separated them
    pub separation_sets: HashMap<(LaggedVar, LaggedVar), Vec<LaggedVar>>,
    /// Number of variables
    pub n_vars: usize,
    /// Maximum lag used
    pub tau_max: usize,
    /// Total number of CI tests performed
    pub n_tests: usize,
}

impl PCStableResult {
    /// Check if there is a link from (source_var, lag) to target_var at lag 0
    pub fn has_link(&self, source_var: usize, lag: usize, target_var: usize) -> bool {
        if let Some(parents) = self.parents.get(&target_var) {
            parents.contains(&(source_var, lag))
        } else {
            false
        }
    }

    /// Get all parents for a given target variable
    pub fn get_parents(&self, target_var: usize) -> &[LaggedVar] {
        static EMPTY: &[LaggedVar] = &[];
        self.parents
            .get(&target_var)
            .map_or(EMPTY, |v| v.as_slice())
    }
}

/// PC-Stable algorithm for time series
pub struct PCStable {
    /// Conditional independence test to use
    ci_test: Box<dyn TimeSeriesCITest>,
    /// Configuration
    config: PCStableConfig,
}

impl PCStable {
    /// Create a new PC-Stable instance
    pub fn new(ci_test: Box<dyn TimeSeriesCITest>, config: PCStableConfig) -> Self {
        Self { ci_test, config }
    }

    /// Run the PC-stable algorithm on multivariate time series data
    ///
    /// # Arguments
    /// * `data` - Multivariate time series of shape (T, n_vars)
    ///
    /// # Returns
    /// `PCStableResult` containing the discovered skeleton and separation sets
    pub fn run(&self, data: &Array2<f64>) -> CausalityResult<PCStableResult> {
        let n_vars = data.ncols();
        let t = data.nrows();

        if t < self.config.tau_max + 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Time series too short for PC-stable with given tau_max".to_string(),
                required: self.config.tau_max + 4,
                actual: t,
            });
        }

        if n_vars == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Data must have at least one variable".to_string(),
            ));
        }

        // Initialize: fully connected lagged graph
        // For each target variable j, all (i, tau) with tau in 1..=tau_max are potential parents
        // Also include (i, 0) for i != j (contemporaneous, for PCMCI+ later)
        let mut adjacencies: HashMap<usize, HashSet<LaggedVar>> = HashMap::new();

        for j in 0..n_vars {
            let mut adj = HashSet::new();
            // Lagged parents
            for tau in 1..=self.config.tau_max {
                for i in 0..n_vars {
                    adj.insert((i, tau));
                }
            }
            // Contemporaneous (for PCMCI+)
            for i in 0..n_vars {
                if i != j {
                    adj.insert((i, 0));
                }
            }
            adjacencies.insert(j, adj);
        }

        let mut separation_sets: HashMap<(LaggedVar, LaggedVar), Vec<LaggedVar>> = HashMap::new();
        let mut n_tests = 0usize;

        // Iterate over conditioning set sizes
        let mut p = 0usize;
        loop {
            let max_p = self.config.max_cond_size.unwrap_or(usize::MAX);
            if p > max_p {
                break;
            }

            let mut any_edge_testable = false;
            // Collect edges to remove (PC-stable: apply all removals after iteration)
            let mut removals: Vec<(usize, LaggedVar, Vec<LaggedVar>)> = Vec::new();

            // Snapshot adjacencies for order independence
            let adj_snapshot: HashMap<usize, Vec<LaggedVar>> = adjacencies
                .iter()
                .map(|(&k, v)| (k, v.iter().copied().collect()))
                .collect();

            for j in 0..n_vars {
                let parents_j: Vec<LaggedVar> =
                    adj_snapshot.get(&j).map_or_else(Vec::new, |v| v.clone());

                for &parent in &parents_j {
                    // Conditioning set candidates: neighbors of j excluding current parent
                    let cond_candidates: Vec<LaggedVar> =
                        parents_j.iter().copied().filter(|&v| v != parent).collect();

                    if cond_candidates.len() < p {
                        continue;
                    }
                    any_edge_testable = true;

                    // Test all subsets of size p
                    let mut found_independent = false;
                    let mut best_sep_set = Vec::new();

                    let subsets = combinations(&cond_candidates, p);
                    for subset in &subsets {
                        n_tests += 1;
                        let result =
                            self.ci_test
                                .test(data, parent, (j, 0), subset, self.config.alpha)?;

                        if !result.dependent {
                            found_independent = true;
                            best_sep_set = subset.clone();
                            break;
                        }
                    }

                    if found_independent {
                        removals.push((j, parent, best_sep_set));
                    }
                }
            }

            // Apply removals (PC-stable: all at once)
            for (target, parent, sep_set) in removals {
                if let Some(adj) = adjacencies.get_mut(&target) {
                    adj.remove(&parent);
                }
                separation_sets.insert((parent, (target, 0)), sep_set);
            }

            if !any_edge_testable {
                break;
            }
            p += 1;
        }

        // Convert adjacencies to result
        let parents: HashMap<usize, Vec<LaggedVar>> = adjacencies
            .into_iter()
            .map(|(k, v)| {
                let mut parents_vec: Vec<LaggedVar> = v.into_iter().collect();
                parents_vec.sort();
                (k, parents_vec)
            })
            .collect();

        Ok(PCStableResult {
            parents,
            separation_sets,
            n_vars,
            tau_max: self.config.tau_max,
            n_tests,
        })
    }
}

/// Generate all combinations of `k` elements from `items`
fn combinations(items: &[LaggedVar], k: usize) -> Vec<Vec<LaggedVar>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > items.len() {
        return vec![];
    }
    if k == items.len() {
        return vec![items.to_vec()];
    }

    let mut result = Vec::new();
    combinations_recursive(items, k, 0, &mut vec![], &mut result);
    result
}

fn combinations_recursive(
    items: &[LaggedVar],
    k: usize,
    start: usize,
    current: &mut Vec<LaggedVar>,
    result: &mut Vec<Vec<LaggedVar>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    let remaining = k - current.len();
    let available = items.len() - start;
    if available < remaining {
        return;
    }
    for i in start..items.len() {
        current.push(items[i]);
        combinations_recursive(items, k, i + 1, current, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causality::ci_tests::ParCorr;
    use scirs2_core::ndarray::Array2;

    fn generate_chain_data(n: usize, seed: u64) -> Array2<f64> {
        // X -> Y -> Z chain with lag 1
        // X_t = 0.7 * X_{t-1} + noise
        // Y_t = 0.5 * X_{t-1} + 0.2 * Y_{t-1} + noise
        // Z_t = 0.5 * Y_{t-1} + 0.2 * Z_{t-1} + noise
        let mut data = Array2::zeros((n, 3));
        let mut state = seed;
        let next_rand = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 32) as f64) / (u32::MAX as f64) - 0.5
        };

        for t in 1..n {
            let e1 = next_rand(&mut state) * 0.1;
            let e2 = next_rand(&mut state) * 0.1;
            let e3 = next_rand(&mut state) * 0.1;
            data[[t, 0]] = 0.7 * data[[t - 1, 0]] + e1;
            data[[t, 1]] = 0.5 * data[[t - 1, 0]] + 0.2 * data[[t - 1, 1]] + e2;
            data[[t, 2]] = 0.5 * data[[t - 1, 1]] + 0.2 * data[[t - 1, 2]] + e3;
        }
        data
    }

    #[test]
    fn test_pc_stable_simple_var1() {
        // 2-variable VAR(1): X_{t-1} -> Y_t
        let n = 500;
        let mut data = Array2::zeros((n, 2));
        let mut state: u64 = 42;
        let next_rand = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 32) as f64) / (u32::MAX as f64) - 0.5
        };
        for t in 1..n {
            let e1 = next_rand(&mut state) * 0.1;
            let e2 = next_rand(&mut state) * 0.1;
            data[[t, 0]] = 0.5 * data[[t - 1, 0]] + e1;
            data[[t, 1]] = 0.4 * data[[t - 1, 0]] + 0.2 * data[[t - 1, 1]] + e2;
        }

        let config = PCStableConfig {
            tau_max: 2,
            alpha: 0.05,
            max_cond_size: Some(2),
        };
        let pc = PCStable::new(Box::new(ParCorr::new()), config);
        let result = pc.run(&data).expect("PC-stable failed");

        // X_{t-1} -> Y_t should be detected
        assert!(result.has_link(0, 1, 1), "Should detect X_{{t-1}} -> Y_t");
    }

    #[test]
    fn test_pc_stable_chain() {
        let data = generate_chain_data(800, 123);
        let config = PCStableConfig {
            tau_max: 2,
            alpha: 0.05,
            max_cond_size: Some(2),
        };
        let pc = PCStable::new(Box::new(ParCorr::new()), config);
        let result = pc.run(&data).expect("PC-stable failed");

        // X_{t-1} -> Y_t should exist
        assert!(
            result.has_link(0, 1, 1),
            "Should detect X_{{t-1}} -> Y_t in chain"
        );
        // Y_{t-1} -> Z_t should exist
        assert!(
            result.has_link(1, 1, 2),
            "Should detect Y_{{t-1}} -> Z_t in chain"
        );
    }

    #[test]
    fn test_pc_stable_separation_sets() {
        let data = generate_chain_data(500, 99);
        let config = PCStableConfig {
            tau_max: 1,
            alpha: 0.05,
            max_cond_size: Some(2),
        };
        let pc = PCStable::new(Box::new(ParCorr::new()), config);
        let result = pc.run(&data).expect("PC-stable failed");

        assert!(result.n_tests > 0, "Should have performed CI tests");
        assert_eq!(result.n_vars, 3);
        assert_eq!(result.tau_max, 1);
    }

    #[test]
    fn test_pc_stable_self_loops() {
        // Each variable should detect its own autoregressive parent
        let n = 500;
        let mut data = Array2::zeros((n, 2));
        let mut state: u64 = 77;
        let next_rand = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 32) as f64) / (u32::MAX as f64) - 0.5
        };
        for t in 1..n {
            data[[t, 0]] = 0.8 * data[[t - 1, 0]] + next_rand(&mut state) * 0.1;
            data[[t, 1]] = 0.7 * data[[t - 1, 1]] + next_rand(&mut state) * 0.1;
        }

        let config = PCStableConfig {
            tau_max: 1,
            alpha: 0.05,
            max_cond_size: Some(1),
        };
        let pc = PCStable::new(Box::new(ParCorr::new()), config);
        let result = pc.run(&data).expect("PC-stable failed");

        // Each variable should have its own lag-1 as parent
        assert!(
            result.has_link(0, 1, 0),
            "X should have autoregressive parent X_{{t-1}}"
        );
        assert!(
            result.has_link(1, 1, 1),
            "Y should have autoregressive parent Y_{{t-1}}"
        );
    }

    #[test]
    fn test_pc_stable_insufficient_data() {
        let data = Array2::zeros((3, 2));
        let config = PCStableConfig {
            tau_max: 2,
            alpha: 0.05,
            max_cond_size: None,
        };
        let pc = PCStable::new(Box::new(ParCorr::new()), config);
        let result = pc.run(&data);
        assert!(result.is_err(), "Should fail with insufficient data");
    }
}
