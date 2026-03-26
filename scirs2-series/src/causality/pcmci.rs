//! PCMCI Algorithm for Causal Discovery in Time Series
//!
//! PCMCI is a two-stage algorithm for discovering causal relationships in
//! multivariate time series data:
//!
//! 1. **Stage 1 (PC-stable)**: Condition selection — identify relevant parents for
//!    each variable using the PC-stable algorithm adapted for time series.
//! 2. **Stage 2 (MCI — Momentary Conditional Independence)**: Test each potential
//!    causal link X_{t-tau} -> Y_t, conditioning on the parents of both X and Y
//!    discovered in Stage 1.
//!
//! The MCI step controls for common drivers and autocorrelation effects, providing
//! more reliable causal inference than pure PC or Granger causality tests.
//!
//! ## PCMCI+ Variant
//!
//! PCMCI+ additionally discovers contemporaneous causal links (tau=0) by including
//! them in the condition selection and MCI stages.
//!
//! ## References
//!
//! - Runge et al. (2019). "Detecting and quantifying causal associations in large
//!   nonlinear time series datasets." Science Advances, 5(11).
//! - Runge (2020). "Discovering contemporaneous and lagged causal relations in
//!   autocorrelated nonlinear time series datasets." UAI 2020.

use std::collections::HashMap;

use crate::error::TimeSeriesError;
use scirs2_core::ndarray::Array2;

use super::ci_tests::{LaggedVar, TimeSeriesCITest};
use super::pc_stable::{PCStable, PCStableConfig, PCStableResult};
use super::CausalityResult;

/// A discovered causal link between two variables
#[derive(Debug, Clone)]
pub struct CausalLink {
    /// Source variable index
    pub source: usize,
    /// Target variable index
    pub target: usize,
    /// Time lag (tau >= 0; tau=0 means contemporaneous)
    pub lag: usize,
    /// Strength of the link (test statistic, e.g. partial correlation)
    pub strength: f64,
    /// p-value for the link
    pub p_value: f64,
}

/// Causal graph represented as an adjacency tensor for time series
#[derive(Debug, Clone)]
pub struct CausalGraph {
    /// Number of variables
    pub n_vars: usize,
    /// Maximum lag
    pub tau_max: usize,
    /// Value matrix: `val_matrix[i][j][tau]` = strength of link `i_{t-tau} -> j_t`
    pub val_matrix: Vec<Vec<Vec<f64>>>,
    /// P-value matrix: `p_matrix[i][j][tau]` = p-value of link `i_{t-tau} -> j_t`
    pub p_matrix: Vec<Vec<Vec<f64>>>,
    /// List of significant causal links
    pub links: Vec<CausalLink>,
}

impl CausalGraph {
    /// Create a new empty causal graph
    fn new(n_vars: usize, tau_max: usize) -> Self {
        let val_matrix = vec![vec![vec![0.0; tau_max + 1]; n_vars]; n_vars];
        let p_matrix = vec![vec![vec![1.0; tau_max + 1]; n_vars]; n_vars];
        Self {
            n_vars,
            tau_max,
            val_matrix,
            p_matrix,
            links: Vec::new(),
        }
    }

    /// Get the strength of link from source at lag to target
    pub fn get_link_strength(&self, source: usize, target: usize, lag: usize) -> f64 {
        if source < self.n_vars && target < self.n_vars && lag <= self.tau_max {
            self.val_matrix[source][target][lag]
        } else {
            0.0
        }
    }

    /// Get the p-value of link from source at lag to target
    pub fn get_link_pvalue(&self, source: usize, target: usize, lag: usize) -> f64 {
        if source < self.n_vars && target < self.n_vars && lag <= self.tau_max {
            self.p_matrix[source][target][lag]
        } else {
            1.0
        }
    }

    /// Check if a link is significant at the given alpha level
    pub fn is_significant(&self, source: usize, target: usize, lag: usize, alpha: f64) -> bool {
        self.get_link_pvalue(source, target, lag) < alpha
    }
}

/// Multiple testing correction method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionMethod {
    /// No correction
    None,
    /// Bonferroni correction
    Bonferroni,
    /// Benjamini-Hochberg FDR
    FDR,
}

/// Configuration for the PCMCI algorithm
#[derive(Debug, Clone)]
pub struct PCMCIConfig {
    /// Maximum lag to consider
    pub tau_max: usize,
    /// Significance level for the PC-stable condition selection stage
    pub alpha_pc: f64,
    /// Significance level for the MCI stage
    pub alpha_mci: f64,
    /// Maximum conditioning set size in PC stage
    pub max_cond_size_pc: Option<usize>,
    /// p-value correction method for MCI stage
    pub correction_method: CorrectionMethod,
    /// Whether to include contemporaneous links (PCMCI+ mode)
    pub include_contemporaneous: bool,
}

impl Default for PCMCIConfig {
    fn default() -> Self {
        Self {
            tau_max: 2,
            alpha_pc: 0.2,
            alpha_mci: 0.05,
            max_cond_size_pc: None,
            correction_method: CorrectionMethod::FDR,
            include_contemporaneous: false,
        }
    }
}

/// Result of the PCMCI algorithm
#[derive(Debug, Clone)]
pub struct PCMCIResult {
    /// The discovered causal graph
    pub graph: CausalGraph,
    /// Parents discovered by the PC-stable stage (before MCI)
    pub pc_parents: HashMap<usize, Vec<LaggedVar>>,
    /// Number of CI tests in PC stage
    pub n_tests_pc: usize,
    /// Number of CI tests in MCI stage
    pub n_tests_mci: usize,
}

/// PCMCI algorithm for causal discovery in time series
pub struct PCMCI {
    /// Conditional independence test
    ci_test: Box<dyn TimeSeriesCITest>,
    /// Algorithm configuration
    config: PCMCIConfig,
}

impl PCMCI {
    /// Create a new PCMCI instance
    pub fn new(ci_test: Box<dyn TimeSeriesCITest>, config: PCMCIConfig) -> Self {
        Self { ci_test, config }
    }

    /// Run the full PCMCI algorithm
    ///
    /// # Arguments
    /// * `data` - Multivariate time series of shape (T, n_vars)
    ///
    /// # Returns
    /// `PCMCIResult` with the discovered causal graph and diagnostics
    pub fn run(&self, data: &Array2<f64>) -> CausalityResult<PCMCIResult> {
        let n_vars = data.ncols();
        let t = data.nrows();

        if n_vars == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Data must have at least one variable".to_string(),
            ));
        }
        if t < self.config.tau_max + 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Time series too short for PCMCI".to_string(),
                required: self.config.tau_max + 4,
                actual: t,
            });
        }

        // ---- Stage 1: PC-stable condition selection ----
        let pc_result = self.run_pc_stage(data)?;

        // ---- Stage 2: MCI tests ----
        let (graph, n_tests_mci) = self.run_mci_stage(data, &pc_result)?;

        Ok(PCMCIResult {
            graph,
            pc_parents: pc_result.parents.clone(),
            n_tests_pc: pc_result.n_tests,
            n_tests_mci,
        })
    }

    /// Run only the PC-stable stage (condition selection)
    fn run_pc_stage(&self, data: &Array2<f64>) -> CausalityResult<PCStableResult> {
        // We need a separate CI test instance for PC stage.
        // Since we can't clone the trait object, we create a fresh ParCorr.
        // The actual CI test is used for MCI stage.
        let pc_config = PCStableConfig {
            tau_max: self.config.tau_max,
            alpha: self.config.alpha_pc,
            max_cond_size: self.config.max_cond_size_pc,
        };

        // For PC stage, use a ParCorr test (standard approach)
        let pc_test = Box::new(super::ci_tests::ParCorr::new());
        let pc = PCStable::new(pc_test, pc_config);
        pc.run(data)
    }

    /// Run the MCI (Momentary Conditional Independence) stage
    ///
    /// For each potential link X_{t-tau} -> Y_t:
    ///   Test X_{t-tau} _||_ Y_t | parents(Y_t) \ {X_{t-tau}} ∪ parents(X_{t-tau_x})
    fn run_mci_stage(
        &self,
        data: &Array2<f64>,
        pc_result: &PCStableResult,
    ) -> CausalityResult<(CausalGraph, usize)> {
        let n_vars = data.ncols();
        let mut graph = CausalGraph::new(n_vars, self.config.tau_max);
        let mut n_tests = 0usize;

        // Collect all (statistic, p_value, source, target, lag) for correction
        let mut test_results: Vec<(f64, f64, usize, usize, usize)> = Vec::new();

        let tau_min = if self.config.include_contemporaneous {
            0
        } else {
            1
        };

        for target in 0..n_vars {
            for source in 0..n_vars {
                for tau in tau_min..=self.config.tau_max {
                    // Skip self-contemporaneous
                    if tau == 0 && source == target {
                        continue;
                    }

                    // Build MCI conditioning set:
                    // parents(target) \ {(source, tau)} ∪ parents(source)
                    let mut cond_set: Vec<LaggedVar> = Vec::new();

                    // Add parents of target (excluding the link being tested)
                    if let Some(target_parents) = pc_result.parents.get(&target) {
                        for &parent in target_parents {
                            if parent != (source, tau) {
                                cond_set.push(parent);
                            }
                        }
                    }

                    // Add parents of source (shifted by tau to align with target's time frame)
                    // parents(X) at time t-tau: if X has parent (k, tau_k),
                    // then in the target's frame it becomes (k, tau + tau_k)
                    if let Some(source_parents) = pc_result.parents.get(&source) {
                        for &(par_var, par_tau) in source_parents {
                            let shifted_lag = tau + par_tau;
                            if shifted_lag <= self.config.tau_max + self.config.tau_max {
                                let shifted_parent = (par_var, shifted_lag);
                                // Exclude the link being tested and the target variable
                                // at lag 0 (we cannot condition on the response itself)
                                if !cond_set.contains(&shifted_parent)
                                    && shifted_parent != (source, tau)
                                    && shifted_parent != (target, 0)
                                {
                                    cond_set.push(shifted_parent);
                                }
                            }
                        }
                    }

                    // Ensure shifted lags don't exceed data bounds
                    let max_allowed_lag = data.nrows().saturating_sub(4);
                    cond_set.retain(|&(_, lag)| lag <= max_allowed_lag);

                    n_tests += 1;
                    let result = self.ci_test.test(
                        data,
                        (source, tau),
                        (target, 0),
                        &cond_set,
                        self.config.alpha_mci,
                    )?;

                    graph.val_matrix[source][target][tau] = result.statistic;
                    graph.p_matrix[source][target][tau] = result.p_value;

                    test_results.push((result.statistic, result.p_value, source, target, tau));
                }
            }
        }

        // Apply multiple testing correction
        match self.config.correction_method {
            CorrectionMethod::None => {}
            CorrectionMethod::Bonferroni => {
                let m = test_results.len() as f64;
                for &(_, p_val, src, tgt, tau) in &test_results {
                    let corrected = (p_val * m).min(1.0);
                    graph.p_matrix[src][tgt][tau] = corrected;
                }
            }
            CorrectionMethod::FDR => {
                apply_fdr_correction(&mut graph, &test_results);
            }
        }

        // Collect significant links
        for &(stat, _, src, tgt, tau) in &test_results {
            let corrected_p = graph.p_matrix[src][tgt][tau];
            if corrected_p < self.config.alpha_mci {
                graph.links.push(CausalLink {
                    source: src,
                    target: tgt,
                    lag: tau,
                    strength: stat,
                    p_value: corrected_p,
                });
            }
        }

        // Sort links by strength (absolute value, descending)
        graph.links.sort_by(|a, b| {
            b.strength
                .abs()
                .partial_cmp(&a.strength.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok((graph, n_tests))
    }
}

/// Apply Benjamini-Hochberg FDR correction to p-values in the graph
fn apply_fdr_correction(graph: &mut CausalGraph, test_results: &[(f64, f64, usize, usize, usize)]) {
    if test_results.is_empty() {
        return;
    }

    let m = test_results.len();

    // Sort by p-value
    let mut indexed: Vec<(usize, f64)> = test_results
        .iter()
        .enumerate()
        .map(|(idx, &(_, p, _, _, _))| (idx, p))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // BH procedure: adjusted_p[k] = min(p[k] * m / rank, 1.0)
    // Then enforce monotonicity from the bottom
    let mut adjusted = vec![0.0f64; m];
    for (rank_minus_1, &(orig_idx, p_val)) in indexed.iter().enumerate() {
        let rank = rank_minus_1 + 1;
        adjusted[orig_idx] = (p_val * m as f64 / rank as f64).min(1.0);
    }

    // Enforce monotonicity: step from largest to smallest p-value
    // adjusted[i] = min(adjusted[i], adjusted[i+1]) for decreasing rank
    let mut prev_adj = 1.0f64;
    for &(orig_idx, _) in indexed.iter().rev() {
        adjusted[orig_idx] = adjusted[orig_idx].min(prev_adj);
        prev_adj = adjusted[orig_idx];
    }

    // Write back to graph
    for (idx, &(_, _, src, tgt, tau)) in test_results.iter().enumerate() {
        graph.p_matrix[src][tgt][tau] = adjusted[idx];
    }
}

/// Convenience function to run PCMCI with default settings
///
/// # Arguments
/// * `data` - Multivariate time series of shape (T, n_vars)
/// * `tau_max` - Maximum lag to consider
/// * `alpha` - Significance level for both PC and MCI stages
///
/// # Returns
/// `PCMCIResult` with discovered causal links
pub fn run_pcmci(data: &Array2<f64>, tau_max: usize, alpha: f64) -> CausalityResult<PCMCIResult> {
    let config = PCMCIConfig {
        tau_max,
        alpha_pc: alpha.max(0.1), // PC stage typically uses a more liberal alpha
        alpha_mci: alpha,
        correction_method: CorrectionMethod::FDR,
        include_contemporaneous: false,
        ..Default::default()
    };
    let ci_test = Box::new(super::ci_tests::ParCorr::new());
    let pcmci = PCMCI::new(ci_test, config);
    pcmci.run(data)
}

/// Convenience function to run PCMCI+ (with contemporaneous links)
///
/// # Arguments
/// * `data` - Multivariate time series of shape (T, n_vars)
/// * `tau_max` - Maximum lag to consider
/// * `alpha` - Significance level
///
/// # Returns
/// `PCMCIResult` with discovered causal links (including contemporaneous)
pub fn run_pcmci_plus(
    data: &Array2<f64>,
    tau_max: usize,
    alpha: f64,
) -> CausalityResult<PCMCIResult> {
    let config = PCMCIConfig {
        tau_max,
        alpha_pc: alpha.max(0.1),
        alpha_mci: alpha,
        correction_method: CorrectionMethod::FDR,
        include_contemporaneous: true,
        ..Default::default()
    };
    let ci_test = Box::new(super::ci_tests::ParCorr::new());
    let pcmci = PCMCI::new(ci_test, config);
    pcmci.run(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn simple_prng(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 32) as f64) / (u32::MAX as f64) - 0.5
    }

    fn generate_var1_2var(n: usize, seed: u64) -> Array2<f64> {
        let mut data = Array2::zeros((n, 2));
        let mut state = seed;
        for t in 1..n {
            let e1 = simple_prng(&mut state) * 0.1;
            let e2 = simple_prng(&mut state) * 0.1;
            data[[t, 0]] = 0.5 * data[[t - 1, 0]] + e1;
            data[[t, 1]] = 0.4 * data[[t - 1, 0]] + 0.2 * data[[t - 1, 1]] + e2;
        }
        data
    }

    fn generate_chain_3var(n: usize, seed: u64) -> Array2<f64> {
        // X -> Y -> Z chain with lag 1
        let mut data = Array2::zeros((n, 3));
        let mut state = seed;
        for t in 1..n {
            let e1 = simple_prng(&mut state) * 0.1;
            let e2 = simple_prng(&mut state) * 0.1;
            let e3 = simple_prng(&mut state) * 0.1;
            data[[t, 0]] = 0.7 * data[[t - 1, 0]] + e1;
            data[[t, 1]] = 0.5 * data[[t - 1, 0]] + 0.2 * data[[t - 1, 1]] + e2;
            data[[t, 2]] = 0.5 * data[[t - 1, 1]] + 0.2 * data[[t - 1, 2]] + e3;
        }
        data
    }

    fn generate_common_cause(n: usize, seed: u64) -> Array2<f64> {
        // Common cause: X -> Y and X -> Z (no direct Y-Z link)
        let mut data = Array2::zeros((n, 3));
        let mut state = seed;
        for t in 1..n {
            let e1 = simple_prng(&mut state) * 0.1;
            let e2 = simple_prng(&mut state) * 0.1;
            let e3 = simple_prng(&mut state) * 0.1;
            data[[t, 0]] = 0.7 * data[[t - 1, 0]] + e1;
            data[[t, 1]] = 0.5 * data[[t - 1, 0]] + 0.2 * data[[t - 1, 1]] + e2;
            data[[t, 2]] = 0.5 * data[[t - 1, 0]] + 0.2 * data[[t - 1, 2]] + e3;
        }
        data
    }

    #[test]
    fn test_pcmci_simple_2var() {
        let data = generate_var1_2var(500, 42);
        let result = run_pcmci(&data, 2, 0.05).expect("PCMCI failed");

        // Should detect X_{t-1} -> Y_t
        let has_x_to_y = result
            .graph
            .links
            .iter()
            .any(|link| link.source == 0 && link.target == 1 && link.lag == 1);
        assert!(has_x_to_y, "PCMCI should detect X_{{t-1}} -> Y_t");
    }

    #[test]
    fn test_pcmci_chain_3var() {
        let data = generate_chain_3var(800, 123);
        let result = run_pcmci(&data, 2, 0.05).expect("PCMCI failed");

        // Should detect X_{t-1} -> Y_t
        let x_to_y = result
            .graph
            .links
            .iter()
            .any(|link| link.source == 0 && link.target == 1 && link.lag == 1);
        assert!(x_to_y, "Should detect X_{{t-1}} -> Y_t in chain");

        // Should detect Y_{t-1} -> Z_t
        let y_to_z = result
            .graph
            .links
            .iter()
            .any(|link| link.source == 1 && link.target == 2 && link.lag == 1);
        assert!(y_to_z, "Should detect Y_{{t-1}} -> Z_t in chain");
    }

    #[test]
    fn test_pcmci_common_cause_no_spurious() {
        let data = generate_common_cause(800, 55);
        let config = PCMCIConfig {
            tau_max: 2,
            alpha_pc: 0.2,
            alpha_mci: 0.05,
            correction_method: CorrectionMethod::FDR,
            include_contemporaneous: false,
            max_cond_size_pc: Some(3),
        };
        let ci_test = Box::new(super::super::ci_tests::ParCorr::new());
        let pcmci = PCMCI::new(ci_test, config);
        let result = pcmci.run(&data).expect("PCMCI failed");

        // Should detect X -> Y and X -> Z
        let x_to_y = result
            .graph
            .links
            .iter()
            .any(|link| link.source == 0 && link.target == 1 && link.lag == 1);
        assert!(x_to_y, "Should detect X_{{t-1}} -> Y_t (common cause)");

        let x_to_z = result
            .graph
            .links
            .iter()
            .any(|link| link.source == 0 && link.target == 2 && link.lag == 1);
        assert!(x_to_z, "Should detect X_{{t-1}} -> Z_t (common cause)");

        // After MCI conditioning on parents of both Y and Z (which includes X),
        // the spurious Y -> Z link should be weaker / not significant
        let spurious_y_to_z = result
            .graph
            .links
            .iter()
            .any(|link| link.source == 1 && link.target == 2 && link.lag == 1);
        // This is the key benefit of MCI: no spurious links from common cause
        // (We allow it to exist but check it's weaker than the true links)
        if spurious_y_to_z {
            let y_z_strength = result.graph.get_link_strength(1, 2, 1).abs();
            let x_y_strength = result.graph.get_link_strength(0, 1, 1).abs();
            // The spurious link should be much weaker if MCI works correctly
            assert!(
                y_z_strength < x_y_strength,
                "Spurious Y->Z ({}) should be weaker than true X->Y ({})",
                y_z_strength,
                x_y_strength
            );
        }
    }

    #[test]
    fn test_pcmci_plus_contemporaneous() {
        // Create data with contemporaneous link: X_t -> Y_t (instantaneous)
        let n = 1000;
        let mut data = Array2::zeros((n, 2));
        let mut state: u64 = 42;
        for t in 1..n {
            let e1 = simple_prng(&mut state) * 0.3;
            let e2 = simple_prng(&mut state) * 0.1;
            data[[t, 0]] = 0.3 * data[[t - 1, 0]] + e1;
            // Y depends on X at same time (contemporaneous) with strong coefficient
            data[[t, 1]] = 0.8 * data[[t, 0]] + 0.1 * data[[t - 1, 1]] + e2;
        }

        let config = PCMCIConfig {
            tau_max: 1,
            alpha_pc: 0.4, // Liberal PC alpha for contemporaneous
            alpha_mci: 0.05,
            correction_method: CorrectionMethod::FDR,
            include_contemporaneous: true,
            max_cond_size_pc: Some(2),
        };
        let ci_test = Box::new(super::super::ci_tests::ParCorr::new());
        let pcmci = PCMCI::new(ci_test, config);
        let result = pcmci.run(&data).expect("PCMCI+ failed");

        // Should detect contemporaneous X_t -> Y_t
        let contemp = result
            .graph
            .links
            .iter()
            .any(|link| link.source == 0 && link.target == 1 && link.lag == 0);
        assert!(contemp, "PCMCI+ should detect contemporaneous X_t -> Y_t");
    }

    #[test]
    fn test_pcmci_correction_methods() {
        let data = generate_var1_2var(300, 77);

        // Test with Bonferroni
        let config_bonf = PCMCIConfig {
            tau_max: 1,
            alpha_pc: 0.2,
            alpha_mci: 0.05,
            correction_method: CorrectionMethod::Bonferroni,
            include_contemporaneous: false,
            max_cond_size_pc: Some(2),
        };
        let ci_test = Box::new(super::super::ci_tests::ParCorr::new());
        let pcmci = PCMCI::new(ci_test, config_bonf);
        let result_bonf = pcmci.run(&data).expect("PCMCI Bonferroni failed");

        assert!(result_bonf.n_tests_mci > 0);
        assert!(result_bonf.n_tests_pc > 0);

        // Test with no correction
        let config_none = PCMCIConfig {
            tau_max: 1,
            alpha_pc: 0.2,
            alpha_mci: 0.05,
            correction_method: CorrectionMethod::None,
            include_contemporaneous: false,
            max_cond_size_pc: Some(2),
        };
        let ci_test2 = Box::new(super::super::ci_tests::ParCorr::new());
        let pcmci2 = PCMCI::new(ci_test2, config_none);
        let result_none = pcmci2.run(&data).expect("PCMCI no-correction failed");

        // With no correction, we should have at least as many significant links
        assert!(
            result_none.graph.links.len() >= result_bonf.graph.links.len(),
            "No correction should find >= links than Bonferroni"
        );
    }

    #[test]
    fn test_pcmci_causal_graph_api() {
        let data = generate_var1_2var(500, 42);
        let result = run_pcmci(&data, 2, 0.05).expect("PCMCI failed");
        let graph = &result.graph;

        assert_eq!(graph.n_vars, 2);
        assert_eq!(graph.tau_max, 2);

        // Test API methods
        let strength = graph.get_link_strength(0, 1, 1);
        assert!(strength.is_finite());

        let pval = graph.get_link_pvalue(0, 1, 1);
        assert!(pval >= 0.0 && pval <= 1.0);

        // Out-of-bounds should return defaults
        assert_eq!(graph.get_link_strength(10, 10, 10), 0.0);
        assert_eq!(graph.get_link_pvalue(10, 10, 10), 1.0);
    }

    #[test]
    fn test_pcmci_insufficient_data() {
        let data = Array2::zeros((5, 2));
        let result = run_pcmci(&data, 3, 0.05);
        assert!(result.is_err());
    }
}
