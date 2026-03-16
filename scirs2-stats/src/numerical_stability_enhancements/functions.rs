//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::StatsResult;
use crate::propertybased_validation::ValidationReport;
use scirs2_core::ndarray::{Array1, ArrayBase, ArrayView1, Data, Ix1};
use scirs2_core::random::{Rng, RngExt};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use super::types::{AdvancedNumericalStabilityConfig, AdvancedNumericalStabilityTester, ComprehensiveStabilityResult, EdgeCaseGenerationApproach, NumericalStabilityThoroughness, PrecisionTestingStrategy, StabilityTolerance};

/// Create comprehensive numerical stability tester
#[allow(dead_code)]
pub fn create_advanced_think_numerical_stability_tester() -> AdvancedNumericalStabilityTester {
    let config = AdvancedNumericalStabilityConfig::default();
    AdvancedNumericalStabilityTester::new(config)
}
/// Create fast numerical stability tester for development
#[allow(dead_code)]
pub fn create_fast_numerical_stability_tester() -> AdvancedNumericalStabilityTester {
    let config = AdvancedNumericalStabilityConfig {
        enable_edge_case_testing: true,
        enable_precision_analysis: true,
        enable_invariant_validation: true,
        enable_cancellation_detection: false,
        enable_overflow_monitoring: false,
        enable_condition_analysis: false,
        enable_differentiation_testing: false,
        enable_convergence_testing: false,
        enable_monte_carlo_testing: false,
        enable_regression_testing: false,
        thoroughness_level: NumericalStabilityThoroughness::Basic,
        precision_strategy: PrecisionTestingStrategy::DoublePrecision,
        edge_case_approach: EdgeCaseGenerationApproach::Predefined,
        stability_tolerance: StabilityTolerance::default(),
        test_timeout: Duration::from_secs(60),
        max_convergence_iterations: 1000,
        monte_carlo_samples: 1000,
    };
    AdvancedNumericalStabilityTester::new(config)
}
/// Create exhaustive numerical stability tester for release validation
#[allow(dead_code)]
pub fn create_exhaustive_numerical_stability_tester() -> AdvancedNumericalStabilityTester {
    let config = AdvancedNumericalStabilityConfig {
        enable_edge_case_testing: true,
        enable_precision_analysis: true,
        enable_invariant_validation: true,
        enable_cancellation_detection: true,
        enable_overflow_monitoring: true,
        enable_condition_analysis: true,
        enable_differentiation_testing: true,
        enable_convergence_testing: true,
        enable_monte_carlo_testing: true,
        enable_regression_testing: true,
        thoroughness_level: NumericalStabilityThoroughness::Exhaustive,
        precision_strategy: PrecisionTestingStrategy::MultiPrecision,
        edge_case_approach: EdgeCaseGenerationApproach::Intelligent,
        stability_tolerance: StabilityTolerance {
            absolute_tolerance: 1e-16,
            relative_tolerance: 1e-14,
            condition_number_threshold: 1e15,
            cancellation_threshold: 1e-12,
            convergence_tolerance: 1e-12,
            monte_carlo_confidence_level: 0.99,
        },
        test_timeout: Duration::from_secs(3600),
        max_convergence_iterations: 100000,
        monte_carlo_samples: 1000000,
    };
    AdvancedNumericalStabilityTester::new(config)
}
/// Enhanced numerical stability testing for common statistical functions
#[allow(dead_code)]
pub fn test_statistical_function_stability<F>(
    function_name: &str,
    test_function: F,
    input_ranges: Vec<(f64, f64)>,
) -> StatsResult<ComprehensiveStabilityResult>
where
    F: Fn(&ArrayView1<f64>) -> StatsResult<f64> + Clone + Send + Sync + 'static,
{
    let config = AdvancedNumericalStabilityConfig::default();
    let tester = AdvancedNumericalStabilityTester::new(config);
    let mut comprehensive_result = ComprehensiveStabilityResult::new(
        function_name.to_string(),
    );
    for (min_val, max_val) in input_ranges {
        let testdata = generate_stability_testdata(min_val, max_val, 1000);
        let range_result = tester
            .comprehensive_stability_testing(
                function_name,
                test_function.clone(),
                &testdata,
            )?;
        if comprehensive_result.edge_case_results.is_none() {
            comprehensive_result.edge_case_results = range_result.edge_case_results;
        }
        if comprehensive_result.precision_results.is_none() {
            comprehensive_result.precision_results = range_result.precision_results;
        }
    }
    Ok(comprehensive_result)
}
/// Generate test data for numerical stability testing
#[allow(dead_code)]
fn generate_stability_testdata(min_val: f64, maxval: f64, size: usize) -> Array1<f64> {
    use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Array1::zeros(size);
    for i in 0..size {
        match i % 5 {
            0 => data[i] = rng.random_range(min_val..maxval),
            1 => data[i] = min_val,
            2 => data[i] = maxval,
            3 => data[i] = (min_val + maxval) / 2.0,
            4 => data[i] = rng.random_range(min_val..maxval) * 1e-10,
            _ => unreachable!(),
        }
    }
    data
}
/// Test numerical stability of mean function specifically
#[allow(dead_code)]
pub fn test_mean_stability() -> StatsResult<ComprehensiveStabilityResult> {
    use crate::descriptive::mean;
    let mean_function = |data: &ArrayView1<f64>| mean(data);
    let input_ranges = vec![
        (- 1e6, 1e6), (- 1.0, 1.0), (- 1e-10, 1e-10), (1e10, 1e11), (- 1e11, - 1e10),
    ];
    test_statistical_function_stability("mean", mean_function, input_ranges)
}
/// Test numerical stability of variance function specifically
#[allow(dead_code)]
pub fn test_variance_stability() -> StatsResult<ComprehensiveStabilityResult> {
    use crate::descriptive::var;
    let variance_function = |data: &ArrayView1<f64>| var(data, 1, None);
    let input_ranges = vec![
        (- 1e6, 1e6), (- 1.0, 1.0), (- 1e-10, 1e-10), (0.0, 1e-6), (1e6, 1e7),
    ];
    test_statistical_function_stability("variance", variance_function, input_ranges)
}
/// Test numerical stability of correlation function specifically
#[allow(dead_code)]
pub fn test_correlation_stability() -> StatsResult<ValidationReport> {
    use crate::propertybased_validation::{
        CorrelationBounds, PropertyBasedValidator, PropertyTestConfig,
    };
    let config = PropertyTestConfig {
        test_cases_per_property: 1000,
        seed: 42,
        tolerance: 1e-12,
        test_edge_cases: true,
        test_cross_platform: true,
        test_numerical_stability: true,
    };
    let mut validator = PropertyBasedValidator::new(config);
    validator.test_property(CorrelationBounds)?;
    Ok(validator.generate_validation_report())
}
/// Run comprehensive numerical stability tests for all core statistical functions
#[allow(dead_code)]
pub fn run_comprehensive_statistical_stability_tests() -> StatsResult<
    HashMap<String, ComprehensiveStabilityResult>,
> {
    let mut results = HashMap::new();
    if let Ok(mean_result) = test_mean_stability() {
        results.insert("mean".to_string(), mean_result);
    }
    if let Ok(var_result) = test_variance_stability() {
        results.insert("variance".to_string(), var_result);
    }
    Ok(results)
}
/// Quick numerical stability validation for CI/CD pipelines
#[allow(dead_code)]
pub fn run_quick_stability_validation() -> StatsResult<bool> {
    let results = run_comprehensive_statistical_stability_tests()?;
    let all_stable = results
        .values()
        .all(|result| {
            result
                .edge_case_results
                .as_ref()
                .map(|edge_results| edge_results.failed_cases.is_empty())
                .unwrap_or(true)
        });
    Ok(all_stable)
}
