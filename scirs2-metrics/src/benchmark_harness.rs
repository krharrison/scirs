//! Metrics benchmark harness for validating metric implementations
//!
//! Designed for use by OxiRS and other COOLJAPAN ecosystem projects.
//! Provides standard benchmark datasets and validation utilities for
//! regression, classification, and ranking metric implementations.
//!
//! # Example
//!
//! ```rust
//! use scirs2_metrics::benchmark_harness::{standard_benchmarks, validate_regression_metric};
//!
//! let benchmarks = standard_benchmarks();
//! let results = validate_regression_metric(
//!     |preds, targets| {
//!         // Simple MSE implementation
//!         let n = preds.len() as f64;
//!         preds.iter().zip(targets.iter())
//!             .map(|(p, t)| (p - t).powi(2))
//!             .sum::<f64>() / n
//!     },
//!     &benchmarks,
//!     "mse",
//!     1e-10,
//! ).expect("validate_regression_metric failed");
//! for (name, passed, _diff) in &results {
//!     assert!(passed, "Benchmark '{}' failed", name);
//! }
//! ```

use crate::error::{MetricsError, Result};

/// Standard regression benchmark dataset for metric validation.
///
/// Each benchmark contains predictions, targets, and known-correct
/// metric values (MSE, MAE, R2) computed analytically.
#[derive(Debug, Clone)]
pub struct MetricBenchmark {
    /// Human-readable benchmark name
    pub name: &'static str,
    /// Predicted values
    pub predictions: Vec<f64>,
    /// Target (ground truth) values
    pub targets: Vec<f64>,
    /// Expected mean squared error
    pub expected_mse: f64,
    /// Expected mean absolute error
    pub expected_mae: f64,
    /// Expected R-squared score
    pub expected_r2: f64,
}

/// Classification benchmark dataset for metric validation.
///
/// Each benchmark contains predicted and true class labels with
/// known-correct accuracy, precision, and recall values.
#[derive(Debug, Clone)]
pub struct ClassificationBenchmark {
    /// Human-readable benchmark name
    pub name: &'static str,
    /// Predicted class labels
    pub predictions: Vec<usize>,
    /// True class labels
    pub targets: Vec<usize>,
    /// Expected accuracy
    pub expected_accuracy: f64,
    /// Expected macro-averaged precision
    pub expected_precision_macro: f64,
    /// Expected macro-averaged recall
    pub expected_recall_macro: f64,
}

/// Ranking / information retrieval benchmark dataset for metric validation.
///
/// Each benchmark contains relevance scores with known-correct
/// NDCG and MAP values.
#[derive(Debug, Clone)]
pub struct RankingBenchmark {
    /// Human-readable benchmark name
    pub name: &'static str,
    /// Relevance scores (higher = more relevant), ordered by rank position
    pub relevance_scores: Vec<f64>,
    /// Expected normalized discounted cumulative gain
    pub expected_ndcg: f64,
    /// Expected mean average precision
    pub expected_map: f64,
}

/// Result of a single benchmark validation run.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Name of the benchmark
    pub benchmark_name: String,
    /// Whether the validation passed within tolerance
    pub passed: bool,
    /// Actual value computed by the metric function
    pub actual: f64,
    /// Expected value from the benchmark
    pub expected: f64,
    /// Absolute difference between actual and expected
    pub difference: f64,
}

/// Returns a suite of standard regression benchmarks with analytically
/// computed expected metric values.
///
/// The benchmarks cover:
/// - Perfect prediction (zero error)
/// - Constant prediction at mean (R2 = 0)
/// - Linear offset
/// - Scaled prediction
/// - Noisy prediction
/// - Inverse prediction
/// - Large magnitude values
/// - Near-zero values
pub fn standard_benchmarks() -> Vec<MetricBenchmark> {
    vec![
        // Perfect prediction: all errors are zero
        MetricBenchmark {
            name: "perfect_prediction",
            predictions: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            targets: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            expected_mse: 0.0,
            expected_mae: 0.0,
            expected_r2: 1.0,
        },
        // Constant prediction at the mean: R2 should be exactly 0
        // targets mean = 3.0, variance = 2.0
        // MSE = ((3-1)^2 + (3-2)^2 + (3-3)^2 + (3-4)^2 + (3-5)^2) / 5 = 10/5 = 2.0
        // MAE = (2+1+0+1+2)/5 = 6/5 = 1.2
        MetricBenchmark {
            name: "constant_mean_prediction",
            predictions: vec![3.0, 3.0, 3.0, 3.0, 3.0],
            targets: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            expected_mse: 2.0,
            expected_mae: 1.2,
            expected_r2: 0.0,
        },
        // Linear offset: predictions = targets + 1
        // MSE = 1.0, MAE = 1.0
        // SS_res = 5*1 = 5, SS_tot = 10, R2 = 1 - 5/10 = 0.5
        MetricBenchmark {
            name: "linear_offset_plus_one",
            predictions: vec![2.0, 3.0, 4.0, 5.0, 6.0],
            targets: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            expected_mse: 1.0,
            expected_mae: 1.0,
            expected_r2: 0.5,
        },
        // Scaled prediction: predictions = 2 * targets
        // targets: [1,2,3,4,5], predictions: [2,4,6,8,10]
        // errors: [1,2,3,4,5]
        // MSE = (1+4+9+16+25)/5 = 55/5 = 11.0
        // MAE = (1+2+3+4+5)/5 = 15/5 = 3.0
        // SS_res = 55, SS_tot = 10, R2 = 1 - 55/10 = -4.5
        MetricBenchmark {
            name: "scaled_double",
            predictions: vec![2.0, 4.0, 6.0, 8.0, 10.0],
            targets: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            expected_mse: 11.0,
            expected_mae: 3.0,
            expected_r2: -4.5,
        },
        // Small symmetric noise: targets = [10, 20, 30, 40, 50]
        // predictions = [10.1, 19.9, 30.1, 39.9, 50.1]
        // errors = [0.1, -0.1, 0.1, -0.1, 0.1]
        // MSE = (0.01 + 0.01 + 0.01 + 0.01 + 0.01)/5 = 0.05/5 = 0.01
        // MAE = (0.1+0.1+0.1+0.1+0.1)/5 = 0.5/5 = 0.1
        // SS_tot for targets = var * n = 200 * 5 = ... let's compute:
        // mean = 30, deviations: [-20,-10,0,10,20], SS_tot = 400+100+0+100+400 = 1000
        // R2 = 1 - 0.05/1000 = 0.99995
        MetricBenchmark {
            name: "small_symmetric_noise",
            predictions: vec![10.1, 19.9, 30.1, 39.9, 50.1],
            targets: vec![10.0, 20.0, 30.0, 40.0, 50.0],
            expected_mse: 0.01,
            expected_mae: 0.1,
            expected_r2: 1.0 - 0.05 / 1000.0,
        },
        // Inverse prediction: predictions = 6 - targets
        // targets: [1,2,3,4,5], preds: [5,4,3,2,1]
        // errors: [4,2,0,-2,-4]
        // MSE = (16+4+0+4+16)/5 = 40/5 = 8.0
        // MAE = (4+2+0+2+4)/5 = 12/5 = 2.4
        // SS_tot = 10, R2 = 1 - 40/10 = -3.0
        MetricBenchmark {
            name: "inverse_prediction",
            predictions: vec![5.0, 4.0, 3.0, 2.0, 1.0],
            targets: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            expected_mse: 8.0,
            expected_mae: 2.4,
            expected_r2: -3.0,
        },
        // Large magnitude: tests numerical stability
        // predictions = targets + 0.001 for all
        // MSE = 0.000001, MAE = 0.001
        MetricBenchmark {
            name: "large_magnitude",
            predictions: vec![1e6 + 0.001, 2e6 + 0.001, 3e6 + 0.001],
            targets: vec![1e6, 2e6, 3e6],
            expected_mse: 1e-6,
            expected_mae: 0.001,
            // SS_tot for [1e6, 2e6, 3e6]: mean = 2e6
            // deviations: [-1e6, 0, 1e6], SS_tot = 2e12
            // R2 = 1 - 3e-6 / 2e12 = ~1.0
            expected_r2: 1.0 - 3e-6 / 2e12,
        },
        // Near-zero targets: tests handling of small values
        // targets: [0.001, 0.002, 0.003, 0.004, 0.005]
        // preds:   [0.002, 0.003, 0.004, 0.005, 0.006]
        // errors:  [0.001, 0.001, 0.001, 0.001, 0.001]
        // MSE = 1e-6, MAE = 0.001
        MetricBenchmark {
            name: "near_zero_values",
            predictions: vec![0.002, 0.003, 0.004, 0.005, 0.006],
            targets: vec![0.001, 0.002, 0.003, 0.004, 0.005],
            expected_mse: 1e-6,
            expected_mae: 0.001,
            // mean target = 0.003
            // SS_tot = (0.002^2 + 0.001^2 + 0 + 0.001^2 + 0.002^2) = 1e-5
            // SS_res = 5e-6
            // R2 = 1 - 5e-6 / 1e-5 = 0.5
            expected_r2: 0.5,
        },
    ]
}

/// Returns a suite of classification benchmarks with known-correct metric values.
///
/// The benchmarks cover:
/// - Perfect classification
/// - Random-like classification
/// - Binary imbalanced
/// - Multi-class scenarios
pub fn classification_benchmarks() -> Vec<ClassificationBenchmark> {
    vec![
        // Perfect classification: 100% on everything
        ClassificationBenchmark {
            name: "perfect_classification",
            predictions: vec![0, 1, 2, 0, 1, 2],
            targets: vec![0, 1, 2, 0, 1, 2],
            expected_accuracy: 1.0,
            expected_precision_macro: 1.0,
            expected_recall_macro: 1.0,
        },
        // All wrong: swap each class
        // predictions [1,0,0,1,0,0], targets [0,1,2,0,1,2]
        // accuracy = 0/6 = 0.0
        ClassificationBenchmark {
            name: "all_wrong",
            predictions: vec![1, 0, 0, 1, 0, 0],
            targets: vec![0, 1, 2, 0, 1, 2],
            expected_accuracy: 0.0,
            // class 0: TP=0, FP=4, precision=0; class 1: TP=0, FP=2, precision=0; class 2: TP=0, FP=0, precision=0/0
            // For macro avg with zero-division handling: 0.0
            expected_precision_macro: 0.0,
            expected_recall_macro: 0.0,
        },
        // Binary: 4 out of 6 correct
        // targets:      [0,0,0,1,1,1]
        // predictions:  [0,0,1,0,1,1]
        // accuracy = 4/6 = 2/3
        // class 0: TP=2, FP=1, FN=1 => precision=2/3, recall=2/3
        // class 1: TP=2, FP=1, FN=1 => precision=2/3, recall=2/3
        // macro precision = 2/3, macro recall = 2/3
        ClassificationBenchmark {
            name: "binary_balanced",
            predictions: vec![0, 0, 1, 0, 1, 1],
            targets: vec![0, 0, 0, 1, 1, 1],
            expected_accuracy: 4.0 / 6.0,
            expected_precision_macro: 2.0 / 3.0,
            expected_recall_macro: 2.0 / 3.0,
        },
        // Single class: all predictions and targets are 0
        ClassificationBenchmark {
            name: "single_class",
            predictions: vec![0, 0, 0, 0],
            targets: vec![0, 0, 0, 0],
            expected_accuracy: 1.0,
            expected_precision_macro: 1.0,
            expected_recall_macro: 1.0,
        },
        // Multi-class with partial correctness
        // targets:      [0,1,2,3,0,1,2,3]
        // predictions:  [0,1,2,3,1,0,3,2]
        // correct: 4/8 = 0.5
        ClassificationBenchmark {
            name: "multiclass_half_correct",
            predictions: vec![0, 1, 2, 3, 1, 0, 3, 2],
            targets: vec![0, 1, 2, 3, 0, 1, 2, 3],
            expected_accuracy: 0.5,
            // class 0: TP=1, FP=1, FN=1 => P=0.5, R=0.5
            // class 1: TP=1, FP=1, FN=1 => P=0.5, R=0.5
            // class 2: TP=1, FP=1, FN=1 => P=0.5, R=0.5
            // class 3: TP=1, FP=1, FN=1 => P=0.5, R=0.5
            // macro: 0.5, 0.5
            expected_precision_macro: 0.5,
            expected_recall_macro: 0.5,
        },
    ]
}

/// Returns a suite of ranking benchmarks with known-correct metric values.
///
/// The benchmarks cover:
/// - Perfect ranking (all relevant items at top)
/// - Single relevant item
/// - No relevant items
/// - Alternating relevance
pub fn ranking_benchmarks() -> Vec<RankingBenchmark> {
    vec![
        // Perfect ranking: descending relevance scores
        // DCG = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.2618.. + 0.5 = 4.7618..
        // IDCG = same (already ideal) => NDCG = 1.0
        // For MAP with binary relevance at threshold > 0: all relevant
        // MAP = (1/1 + 2/2 + 3/3)/3 = 1.0
        RankingBenchmark {
            name: "perfect_descending",
            relevance_scores: vec![3.0, 2.0, 1.0],
            expected_ndcg: 1.0,
            expected_map: 1.0,
        },
        // Reverse ranking: ascending relevance
        // DCG = 1/log2(2) + 2/log2(3) + 3/log2(4) = 1 + 1.2618.. + 1.5 = 3.7618..
        // IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.2618.. + 0.5 = 4.7618..
        // NDCG = 3.7618.. / 4.7618.. ≈ 0.7902..
        RankingBenchmark {
            name: "reverse_ranking",
            relevance_scores: vec![1.0, 2.0, 3.0],
            expected_ndcg: {
                let dcg = 1.0 / 2.0_f64.log2() + 2.0 / 3.0_f64.log2() + 3.0 / 4.0_f64.log2();
                let idcg = 3.0 / 2.0_f64.log2() + 2.0 / 3.0_f64.log2() + 1.0 / 4.0_f64.log2();
                dcg / idcg
            },
            // MAP: items at positions 1,2,3 are all relevant (scores > 0)
            // Already all relevant, so MAP = 1.0
            expected_map: 1.0,
        },
        // Single relevant item at position 1 (0-indexed)
        // DCG = 0/log2(2) + 1/log2(3) = 0 + 0.6309..
        // IDCG = 1/log2(2) = 1.0
        // NDCG = 0.6309..
        RankingBenchmark {
            name: "single_relevant_second",
            relevance_scores: vec![0.0, 1.0, 0.0, 0.0],
            expected_ndcg: {
                let dcg = 1.0 / 3.0_f64.log2();
                let idcg = 1.0 / 2.0_f64.log2();
                dcg / idcg
            },
            // MAP: 1 relevant item at position 2 => P@2 = 1/2 = 0.5
            // AP = 0.5, MAP = 0.5
            expected_map: 0.5,
        },
        // No relevant items
        RankingBenchmark {
            name: "no_relevant_items",
            relevance_scores: vec![0.0, 0.0, 0.0],
            expected_ndcg: 0.0,
            expected_map: 0.0,
        },
        // Binary alternating: [1, 0, 1, 0, 1]
        // DCG = 1/log2(2) + 0 + 1/log2(4) + 0 + 1/log2(6)
        //     = 1.0 + 0.5 + 0.38685..  = 1.88685..
        // IDCG (ideal = [1,1,1,0,0]) = 1/log2(2) + 1/log2(3) + 1/log2(4) + 0 + 0
        //     = 1.0 + 0.6309.. + 0.5 = 2.1309..
        // NDCG = 1.88685.. / 2.1309.. ≈ 0.88549..
        RankingBenchmark {
            name: "binary_alternating",
            relevance_scores: vec![1.0, 0.0, 1.0, 0.0, 1.0],
            expected_ndcg: {
                let dcg = 1.0 / 2.0_f64.log2() + 1.0 / 4.0_f64.log2() + 1.0 / 6.0_f64.log2();
                let idcg = 1.0 / 2.0_f64.log2() + 1.0 / 3.0_f64.log2() + 1.0 / 4.0_f64.log2();
                dcg / idcg
            },
            // MAP: relevant at positions 1,3,5
            // P@1 = 1/1, P@3 = 2/3, P@5 = 3/5
            // AP = (1 + 2/3 + 3/5)/3 = (1 + 0.6667 + 0.6)/3 = 2.2667/3 ≈ 0.75556
            expected_map: (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0,
        },
    ]
}

/// Validates a regression metric function against the standard benchmark suite.
///
/// The `metric_fn` receives `(predictions, targets)` slices and should return
/// a single f64 metric value. The `expected_field` selects which benchmark
/// field to compare against: `"mse"`, `"mae"`, or `"r2"`.
///
/// Returns a list of `(benchmark_name, passed, difference)` tuples.
///
/// # Errors
///
/// Returns `MetricsError::InvalidArgument` if `expected_field` is not
/// one of `"mse"`, `"mae"`, or `"r2"`.
///
/// # Example
///
/// ```rust
/// use scirs2_metrics::benchmark_harness::{standard_benchmarks, validate_regression_metric};
///
/// let benchmarks = standard_benchmarks();
/// let results = validate_regression_metric(
///     |preds, targets| {
///         let n = preds.len() as f64;
///         preds.iter().zip(targets.iter())
///             .map(|(p, t)| (p - t).powi(2))
///             .sum::<f64>() / n
///     },
///     &benchmarks,
///     "mse",
///     1e-10,
/// );
/// assert!(results.is_ok());
/// ```
pub fn validate_regression_metric<F: Fn(&[f64], &[f64]) -> f64>(
    metric_fn: F,
    benchmarks: &[MetricBenchmark],
    expected_field: &str,
    tolerance: f64,
) -> Result<Vec<(String, bool, f64)>> {
    // Validate the field selector
    if !matches!(expected_field, "mse" | "mae" | "r2") {
        return Err(MetricsError::InvalidArgument(format!(
            "expected_field must be 'mse', 'mae', or 'r2', got '{}'",
            expected_field
        )));
    }

    let mut results = Vec::with_capacity(benchmarks.len());

    for bench in benchmarks {
        let actual = metric_fn(&bench.predictions, &bench.targets);
        let expected = match expected_field {
            "mse" => bench.expected_mse,
            "mae" => bench.expected_mae,
            "r2" => bench.expected_r2,
            _ => unreachable!(), // already validated above
        };

        let diff = (actual - expected).abs();
        let passed = diff <= tolerance;
        results.push((bench.name.to_string(), passed, diff));
    }

    Ok(results)
}

/// Validates a regression metric using the richer `ValidationResult` output.
///
/// Similar to `validate_regression_metric` but returns full `ValidationResult`
/// structs with all details.
///
/// # Errors
///
/// Returns `MetricsError::InvalidArgument` if `expected_field` is not valid.
pub fn validate_regression_metric_detailed<F: Fn(&[f64], &[f64]) -> f64>(
    metric_fn: F,
    benchmarks: &[MetricBenchmark],
    expected_field: &str,
    tolerance: f64,
) -> Result<Vec<ValidationResult>> {
    if !matches!(expected_field, "mse" | "mae" | "r2") {
        return Err(MetricsError::InvalidArgument(format!(
            "expected_field must be 'mse', 'mae', or 'r2', got '{}'",
            expected_field
        )));
    }

    let mut results = Vec::with_capacity(benchmarks.len());

    for bench in benchmarks {
        let actual = metric_fn(&bench.predictions, &bench.targets);
        let expected = match expected_field {
            "mse" => bench.expected_mse,
            "mae" => bench.expected_mae,
            "r2" => bench.expected_r2,
            _ => unreachable!(),
        };

        let difference = (actual - expected).abs();
        results.push(ValidationResult {
            benchmark_name: bench.name.to_string(),
            passed: difference <= tolerance,
            actual,
            expected,
            difference,
        });
    }

    Ok(results)
}

/// Validates a classification metric function against the classification
/// benchmark suite.
///
/// The `metric_fn` receives `(predictions, targets)` slices and should
/// return a single f64 value. The `expected_field` selects which benchmark
/// field to compare: `"accuracy"`, `"precision_macro"`, or `"recall_macro"`.
///
/// # Errors
///
/// Returns `MetricsError::InvalidArgument` if `expected_field` is invalid.
pub fn validate_classification_metric<F: Fn(&[usize], &[usize]) -> f64>(
    metric_fn: F,
    benchmarks: &[ClassificationBenchmark],
    expected_field: &str,
    tolerance: f64,
) -> Result<Vec<(String, bool, f64)>> {
    if !matches!(
        expected_field,
        "accuracy" | "precision_macro" | "recall_macro"
    ) {
        return Err(MetricsError::InvalidArgument(format!(
            "expected_field must be 'accuracy', 'precision_macro', or 'recall_macro', got '{}'",
            expected_field
        )));
    }

    let mut results = Vec::with_capacity(benchmarks.len());

    for bench in benchmarks {
        let actual = metric_fn(&bench.predictions, &bench.targets);
        let expected = match expected_field {
            "accuracy" => bench.expected_accuracy,
            "precision_macro" => bench.expected_precision_macro,
            "recall_macro" => bench.expected_recall_macro,
            _ => unreachable!(),
        };

        let diff = (actual - expected).abs();
        let passed = diff <= tolerance;
        results.push((bench.name.to_string(), passed, diff));
    }

    Ok(results)
}

/// Validates a ranking metric function against the ranking benchmark suite.
///
/// The `metric_fn` receives a relevance scores slice and should return
/// a single f64 value. The `expected_field` selects which benchmark
/// field to compare: `"ndcg"` or `"map"`.
///
/// # Errors
///
/// Returns `MetricsError::InvalidArgument` if `expected_field` is invalid.
pub fn validate_ranking_metric<F: Fn(&[f64]) -> f64>(
    metric_fn: F,
    benchmarks: &[RankingBenchmark],
    expected_field: &str,
    tolerance: f64,
) -> Result<Vec<(String, bool, f64)>> {
    if !matches!(expected_field, "ndcg" | "map") {
        return Err(MetricsError::InvalidArgument(format!(
            "expected_field must be 'ndcg' or 'map', got '{}'",
            expected_field
        )));
    }

    let mut results = Vec::with_capacity(benchmarks.len());

    for bench in benchmarks {
        let actual = metric_fn(&bench.relevance_scores);
        let expected = match expected_field {
            "ndcg" => bench.expected_ndcg,
            "map" => bench.expected_map,
            _ => unreachable!(),
        };

        let diff = (actual - expected).abs();
        let passed = diff <= tolerance;
        results.push((bench.name.to_string(), passed, diff));
    }

    Ok(results)
}

/// Checks consistency of all benchmarks: predictions and targets must have
/// equal length, and all expected values must be finite.
///
/// Returns `Ok(())` if all benchmarks are consistent, or an error describing
/// the first inconsistency found.
pub fn check_benchmark_consistency(benchmarks: &[MetricBenchmark]) -> Result<()> {
    for bench in benchmarks {
        if bench.predictions.len() != bench.targets.len() {
            return Err(MetricsError::InvalidInput(format!(
                "Benchmark '{}': predictions length ({}) != targets length ({})",
                bench.name,
                bench.predictions.len(),
                bench.targets.len()
            )));
        }
        if bench.predictions.is_empty() {
            return Err(MetricsError::InvalidInput(format!(
                "Benchmark '{}': empty predictions/targets",
                bench.name
            )));
        }
        if !bench.expected_mse.is_finite() {
            return Err(MetricsError::InvalidInput(format!(
                "Benchmark '{}': expected_mse is not finite",
                bench.name
            )));
        }
        if !bench.expected_mae.is_finite() {
            return Err(MetricsError::InvalidInput(format!(
                "Benchmark '{}': expected_mae is not finite",
                bench.name
            )));
        }
        if !bench.expected_r2.is_finite() {
            return Err(MetricsError::InvalidInput(format!(
                "Benchmark '{}': expected_r2 is not finite",
                bench.name
            )));
        }
    }
    Ok(())
}

/// Checks consistency of classification benchmarks.
pub fn check_classification_benchmark_consistency(
    benchmarks: &[ClassificationBenchmark],
) -> Result<()> {
    for bench in benchmarks {
        if bench.predictions.len() != bench.targets.len() {
            return Err(MetricsError::InvalidInput(format!(
                "Benchmark '{}': predictions length ({}) != targets length ({})",
                bench.name,
                bench.predictions.len(),
                bench.targets.len()
            )));
        }
        if bench.predictions.is_empty() {
            return Err(MetricsError::InvalidInput(format!(
                "Benchmark '{}': empty predictions/targets",
                bench.name
            )));
        }
        if !bench.expected_accuracy.is_finite()
            || !bench.expected_precision_macro.is_finite()
            || !bench.expected_recall_macro.is_finite()
        {
            return Err(MetricsError::InvalidInput(format!(
                "Benchmark '{}': expected metric values must be finite",
                bench.name
            )));
        }
    }
    Ok(())
}

/// Checks consistency of ranking benchmarks.
pub fn check_ranking_benchmark_consistency(benchmarks: &[RankingBenchmark]) -> Result<()> {
    for bench in benchmarks {
        if bench.relevance_scores.is_empty() {
            return Err(MetricsError::InvalidInput(format!(
                "Benchmark '{}': empty relevance scores",
                bench.name
            )));
        }
        if !bench.expected_ndcg.is_finite() || !bench.expected_map.is_finite() {
            return Err(MetricsError::InvalidInput(format!(
                "Benchmark '{}': expected metric values must be finite",
                bench.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_benchmarks_not_empty() {
        let benchmarks = standard_benchmarks();
        assert!(!benchmarks.is_empty());
        assert!(benchmarks.len() >= 8, "Expected at least 8 benchmarks");
    }

    #[test]
    fn test_perfect_prediction_mse_zero() {
        let benchmarks = standard_benchmarks();
        let perfect = benchmarks
            .iter()
            .find(|b| b.name == "perfect_prediction")
            .expect("perfect_prediction benchmark should exist");
        assert!((perfect.expected_mse - 0.0).abs() < 1e-15);
        assert!((perfect.expected_mae - 0.0).abs() < 1e-15);
        assert!((perfect.expected_r2 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_constant_prediction_r2_zero() {
        let benchmarks = standard_benchmarks();
        let constant = benchmarks
            .iter()
            .find(|b| b.name == "constant_mean_prediction")
            .expect("constant_mean_prediction benchmark should exist");
        assert!(
            (constant.expected_r2 - 0.0).abs() < 1e-15,
            "R2 for constant mean prediction should be 0"
        );
        assert!(
            (constant.expected_mse - 2.0).abs() < 1e-15,
            "MSE should be 2.0"
        );
    }

    #[test]
    fn test_classification_benchmarks_not_empty() {
        let benchmarks = classification_benchmarks();
        assert!(!benchmarks.is_empty());
        assert!(benchmarks.len() >= 4);
    }

    #[test]
    fn test_validate_regression_metric_perfect() {
        let benchmarks = standard_benchmarks();

        // A correct MSE implementation
        let mse_fn = |preds: &[f64], targets: &[f64]| -> f64 {
            let n = preds.len() as f64;
            preds
                .iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / n
        };

        let results = validate_regression_metric(mse_fn, &benchmarks, "mse", 1e-8)
            .expect("validation should succeed");
        assert!(!results.is_empty());

        // Check the perfect prediction case passes
        let perfect_result = results
            .iter()
            .find(|(name, _, _)| name == "perfect_prediction")
            .expect("should find perfect_prediction");
        assert!(perfect_result.1, "Perfect prediction MSE should pass");
    }

    #[test]
    fn test_validate_regression_metric_with_known_bad() {
        let benchmarks = standard_benchmarks();

        // A deliberately wrong metric: always returns 42.0
        let bad_fn = |_preds: &[f64], _targets: &[f64]| -> f64 { 42.0 };

        let results = validate_regression_metric(bad_fn, &benchmarks, "mse", 1e-10)
            .expect("validation should succeed");

        // Most benchmarks should fail
        let failures: Vec<_> = results.iter().filter(|(_, passed, _)| !passed).collect();
        assert!(
            !failures.is_empty(),
            "A bad metric should fail some benchmarks"
        );
    }

    #[test]
    fn test_validate_regression_metric_invalid_field() {
        let benchmarks = standard_benchmarks();
        let f = |_: &[f64], _: &[f64]| -> f64 { 0.0 };
        let result = validate_regression_metric(f, &benchmarks, "invalid", 1e-10);
        assert!(result.is_err());
    }

    #[test]
    fn test_ranking_benchmarks_not_empty() {
        let benchmarks = ranking_benchmarks();
        assert!(!benchmarks.is_empty());
        assert!(benchmarks.len() >= 4);
    }

    #[test]
    fn test_benchmark_consistency() {
        let reg = standard_benchmarks();
        assert!(check_benchmark_consistency(&reg).is_ok());

        let cls = classification_benchmarks();
        assert!(check_classification_benchmark_consistency(&cls).is_ok());

        let rank = ranking_benchmarks();
        assert!(check_ranking_benchmark_consistency(&rank).is_ok());
    }

    #[test]
    fn test_benchmark_consistency_catches_length_mismatch() {
        let bad_bench = vec![MetricBenchmark {
            name: "bad",
            predictions: vec![1.0, 2.0],
            targets: vec![1.0],
            expected_mse: 0.0,
            expected_mae: 0.0,
            expected_r2: 0.0,
        }];
        assert!(check_benchmark_consistency(&bad_bench).is_err());
    }

    #[test]
    fn test_validate_classification_metric() {
        let benchmarks = classification_benchmarks();

        // Simple accuracy function
        let accuracy_fn = |preds: &[usize], targets: &[usize]| -> f64 {
            let correct = preds
                .iter()
                .zip(targets.iter())
                .filter(|(p, t)| p == t)
                .count();
            correct as f64 / preds.len() as f64
        };

        let results = validate_classification_metric(accuracy_fn, &benchmarks, "accuracy", 1e-10)
            .expect("validation should succeed");
        assert!(!results.is_empty());

        // Perfect classification should pass
        let perfect = results
            .iter()
            .find(|(name, _, _)| name == "perfect_classification")
            .expect("should find perfect_classification");
        assert!(perfect.1, "Perfect classification accuracy should pass");
    }

    #[test]
    fn test_validate_ranking_metric() {
        let benchmarks = ranking_benchmarks();

        // Simple NDCG implementation for verification
        let ndcg_fn = |scores: &[f64]| -> f64 {
            if scores.is_empty() {
                return 0.0;
            }

            // Compute DCG
            let dcg: f64 = scores
                .iter()
                .enumerate()
                .map(|(i, &rel)| rel / ((i + 2) as f64).log2())
                .sum();

            // Compute ideal DCG
            let mut sorted_scores = scores.to_vec();
            sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let idcg: f64 = sorted_scores
                .iter()
                .enumerate()
                .map(|(i, &rel)| rel / ((i + 2) as f64).log2())
                .sum();

            if idcg == 0.0 {
                0.0
            } else {
                dcg / idcg
            }
        };

        let results = validate_ranking_metric(ndcg_fn, &benchmarks, "ndcg", 1e-10)
            .expect("validation should succeed");
        assert!(!results.is_empty());

        // Perfect descending should have NDCG = 1.0
        let perfect = results
            .iter()
            .find(|(name, _, _)| name == "perfect_descending")
            .expect("should find perfect_descending");
        assert!(perfect.1, "Perfect descending should have NDCG = 1.0");
    }

    #[test]
    fn test_detailed_validation() {
        let benchmarks = standard_benchmarks();
        let mse_fn = |preds: &[f64], targets: &[f64]| -> f64 {
            let n = preds.len() as f64;
            preds
                .iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / n
        };

        let results = validate_regression_metric_detailed(mse_fn, &benchmarks, "mse", 1e-8)
            .expect("should succeed");
        assert!(!results.is_empty());

        for result in &results {
            assert!(result.difference.is_finite());
            assert!(result.expected.is_finite());
            assert!(result.actual.is_finite());
            assert!(!result.benchmark_name.is_empty());
        }
    }
}
