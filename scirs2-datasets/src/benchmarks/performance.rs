//! Performance benchmarking utilities
//!
//! This module provides tools for benchmarking dataset operations against scikit-learn
//! and other reference implementations to measure performance improvements.

use crate::generators::*;
use crate::loaders::{load_csv, load_csv_parallel, CsvConfig, StreamingConfig};
use crate::sample::load_wine;
use crate::toy::{load_diabetes, *};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

/// Benchmark result for a single operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the operation
    pub operation: String,
    /// Parameters used in the benchmark
    pub parameters: HashMap<String, String>,
    /// Total execution time
    pub duration: Duration,
    /// Memory usage in bytes (if measured)
    pub memory_used: Option<usize>,
    /// Number of samples processed
    pub samples: usize,
    /// Number of features
    pub features: usize,
    /// Throughput (samples per second)
    pub throughput: f64,
    /// Success/failure status
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(operation: String, parameters: HashMap<String, String>) -> Self {
        Self {
            operation,
            parameters,
            duration: Duration::ZERO,
            memory_used: None,
            samples: 0,
            features: 0,
            throughput: 0.0,
            success: false,
            error: None,
        }
    }

    /// Mark the benchmark as successful with timing information
    pub fn success(mut self, duration: Duration, samples: usize, features: usize) -> Self {
        self.duration = duration;
        self.samples = samples;
        self.features = features;
        self.throughput = if duration.as_secs_f64() > 0.0 {
            samples as f64 / duration.as_secs_f64()
        } else {
            0.0
        };
        self.success = true;
        self
    }

    /// Mark the benchmark as failed with error message
    pub fn failure(mut self, error: String) -> Self {
        self.success = false;
        self.error = Some(error);
        self
    }

    /// Set memory usage
    pub fn with_memory(mut self, memoryused: usize) -> Self {
        self.memory_used = Some(memoryused);
        self
    }

    /// Get formatted duration string
    pub fn formatted_duration(&self) -> String {
        if self.duration.as_secs() > 0 {
            format!("{:.2}s", self.duration.as_secs_f64())
        } else if self.duration.as_millis() > 0 {
            format!("{}ms", self.duration.as_millis())
        } else {
            format!("{}μs", self.duration.as_micros())
        }
    }

    /// Get formatted throughput string
    pub fn formatted_throughput(&self) -> String {
        if self.throughput >= 1000.0 {
            format!("{:.1}K samples/s", self.throughput / 1000.0)
        } else {
            format!("{:.1} samples/s", self.throughput)
        }
    }

    /// Get formatted memory usage string
    pub fn formatted_memory(&self) -> String {
        match self.memory_used {
            Some(bytes) => {
                if bytes >= 1024 * 1024 * 1024 {
                    format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                } else if bytes >= 1024 * 1024 {
                    format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
                } else if bytes >= 1024 {
                    format!("{:.1} KB", bytes as f64 / 1024.0)
                } else {
                    format!("{bytes} B")
                }
            }
            None => "N/A".to_string(),
        }
    }
}

/// Collection of benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    /// Name of the benchmark suite
    pub name: String,
    /// Individual benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Total time for the entire suite
    pub total_duration: Duration,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(name: String) -> Self {
        Self {
            name,
            results: Vec::new(),
            total_duration: Duration::ZERO,
        }
    }

    /// Add a benchmark result
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.total_duration += result.duration;
        self.results.push(result);
    }

    /// Get successful results only
    pub fn successful_results(&self) -> Vec<&BenchmarkResult> {
        self.results.iter().filter(|r| r.success).collect()
    }

    /// Get failed results only
    pub fn failed_results(&self) -> Vec<&BenchmarkResult> {
        self.results.iter().filter(|r| !r.success).collect()
    }

    /// Calculate average throughput
    pub fn average_throughput(&self) -> f64 {
        let successful = self.successful_results();
        if successful.is_empty() {
            0.0
        } else {
            successful.iter().map(|r| r.throughput).sum::<f64>() / successful.len() as f64
        }
    }

    /// Get total samples processed
    pub fn total_samples(&self) -> usize {
        self.successful_results().iter().map(|r| r.samples).sum()
    }

    /// Print a summary report
    pub fn print_summary(&self) {
        println!("=== Benchmark Suite: {} ===", self.name);
        println!("Total duration: {:.2}s", self.total_duration.as_secs_f64());
        println!(
            "Successful benchmarks: {}/{}",
            self.successful_results().len(),
            self.results.len()
        );
        println!("Total samples processed: {}", self.total_samples());
        println!(
            "Average throughput: {:.1} samples/s",
            self.average_throughput()
        );

        if !self.failed_results().is_empty() {
            println!("\nFailed benchmarks:");
            for result in self.failed_results() {
                println!(
                    "  - {}: {}",
                    result.operation,
                    result
                        .error
                        .as_ref()
                        .unwrap_or(&"Unknown error".to_string())
                );
            }
        }

        println!("\nDetailed results:");
        for result in &self.results {
            if result.success {
                println!(
                    "  {} - {} ({} samples, {} features) - {}",
                    result.operation,
                    result.formatted_duration(),
                    result.samples,
                    result.features,
                    result.formatted_throughput()
                );
            }
        }
    }
}

/// Benchmark runner for dataset operations
pub struct BenchmarkRunner {
    /// Number of iterations for each benchmark
    pub iterations: usize,
    /// Whether to include memory measurements
    pub measure_memory: bool,
    /// Warmup iterations before actual benchmarks
    pub warmup_iterations: usize,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self {
            iterations: 5,
            measure_memory: false,
            warmup_iterations: 1,
        }
    }
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Enable memory measurement
    pub fn with_memory_measurement(mut self, measure: bool) -> Self {
        self.measure_memory = measure;
        self
    }

    /// Set warmup iterations
    pub fn with_warmup(mut self, warmupiterations: usize) -> Self {
        self.warmup_iterations = warmupiterations;
        self
    }

    /// Run a benchmark function multiple times and return average result
    pub fn run_benchmark<F>(
        &self,
        name: &str,
        parameters: HashMap<String, String>,
        mut benchmark_fn: F,
    ) -> BenchmarkResult
    where
        F: FnMut() -> std::result::Result<(usize, usize), String>,
    {
        // Warmup runs
        for _ in 0..self.warmup_iterations {
            let _ = benchmark_fn();
        }

        let mut durations = Vec::new();
        let mut last_samples = 0;
        let mut last_features = 0;
        let mut last_error = None;

        // Actual benchmark runs
        for _ in 0..self.iterations {
            let start = Instant::now();
            match benchmark_fn() {
                Ok((samples, features)) => {
                    let duration = start.elapsed();
                    durations.push(duration);
                    last_samples = samples;
                    last_features = features;
                }
                Err(e) => {
                    last_error = Some(e);
                    break;
                }
            }
        }

        if let Some(error) = last_error {
            return BenchmarkResult::new(name.to_string(), parameters).failure(error);
        }

        if durations.is_empty() {
            return BenchmarkResult::new(name.to_string(), parameters)
                .failure("No successful runs".to_string());
        }

        // Calculate average duration
        let total_duration: Duration = durations.iter().sum();
        let avg_duration = total_duration / durations.len() as u32;

        BenchmarkResult::new(name.to_string(), parameters).success(
            avg_duration,
            last_samples,
            last_features,
        )
    }

    /// Benchmark toy dataset loading
    pub fn benchmark_toy_datasets(&self) -> BenchmarkSuite {
        let mut suite = BenchmarkSuite::new("Toy Datasets".to_string());

        // Benchmark iris dataset
        let iris_params = HashMap::from([("dataset".to_string(), "iris".to_string())]);
        let iris_result = self.run_benchmark("load_iris", iris_params, || match load_iris() {
            Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
            Err(e) => Err(format!("Failed to load iris: {e}")),
        });
        suite.add_result(iris_result);

        // Benchmark boston dataset
        let boston_params = HashMap::from([("dataset".to_string(), "boston".to_string())]);
        let boston_result =
            self.run_benchmark("load_boston", boston_params, || match load_boston() {
                Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
                Err(e) => Err(format!("Failed to load boston: {e}")),
            });
        suite.add_result(boston_result);

        // Benchmark digits dataset
        let digits_params = HashMap::from([("dataset".to_string(), "digits".to_string())]);
        let digits_result =
            self.run_benchmark("load_digits", digits_params, || match load_digits() {
                Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
                Err(e) => Err(format!("Failed to load digits: {e}")),
            });
        suite.add_result(digits_result);

        // Benchmark wine dataset
        let wine_params = HashMap::from([("dataset".to_string(), "wine".to_string())]);
        let wine_result = self.run_benchmark("load_wine", wine_params, || match load_wine(false) {
            Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
            Err(e) => Err(format!("Failed to load wine: {e}")),
        });
        suite.add_result(wine_result);

        // Benchmark breast cancer dataset
        let bc_params = HashMap::from([("dataset".to_string(), "breast_cancer".to_string())]);
        let bc_result =
            self.run_benchmark(
                "load_breast_cancer",
                bc_params,
                || match load_breast_cancer() {
                    Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
                    Err(e) => Err(format!("Failed to load breastcancer: {e}")),
                },
            );
        suite.add_result(bc_result);

        // Benchmark diabetes dataset
        let diabetes_params = HashMap::from([("dataset".to_string(), "diabetes".to_string())]);
        let diabetes_result =
            self.run_benchmark("load_diabetes", diabetes_params, || match load_diabetes() {
                Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
                Err(e) => Err(format!("Failed to load diabetes: {e}")),
            });
        suite.add_result(diabetes_result);

        suite
    }

    /// Benchmark synthetic data generation
    pub fn benchmark_data_generation(&self) -> BenchmarkSuite {
        let mut suite = BenchmarkSuite::new("Data Generation".to_string());

        // Test different dataset sizes
        let sizes = vec![100, 1000, 10000];
        let features = vec![5, 20, 100];

        for &n_samples in &sizes {
            for &n_features in &features {
                // Classification benchmark
                let class_params = HashMap::from([
                    ("type".to_string(), "classification".to_string()),
                    ("samples".to_string(), n_samples.to_string()),
                    ("features".to_string(), n_features.to_string()),
                ]);
                let class_result = self.run_benchmark(
                    &format!("make_classification_{n_samples}x{n_features}"),
                    class_params,
                    || match make_classification(n_samples, n_features, 3, 2, 4, Some(42)) {
                        Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
                        Err(e) => Err(format!("Failed to generate classification data: {e}")),
                    },
                );
                suite.add_result(class_result);

                // Regression benchmark
                let reg_params = HashMap::from([
                    ("type".to_string(), "regression".to_string()),
                    ("samples".to_string(), n_samples.to_string()),
                    ("features".to_string(), n_features.to_string()),
                ]);
                let reg_result = self.run_benchmark(
                    &format!("make_regression_{n_samples}x{n_features}"),
                    reg_params,
                    || match make_regression(n_samples, n_features, 3, 0.1, Some(42)) {
                        Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
                        Err(e) => Err(format!("Failed to generate regression data: {e}")),
                    },
                );
                suite.add_result(reg_result);

                // Clustering benchmark (only for 2D for now)
                if n_features <= 10 {
                    let blob_params = HashMap::from([
                        ("type".to_string(), "blobs".to_string()),
                        ("samples".to_string(), n_samples.to_string()),
                        ("features".to_string(), n_features.to_string()),
                    ]);
                    let blob_result = self.run_benchmark(
                        &format!("make_blobs_{n_samples}x{n_features}"),
                        blob_params,
                        || match make_blobs(n_samples, n_features, 4, 1.0, Some(42)) {
                            Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
                            Err(e) => Err(format!("Failed to generate blob data: {e}")),
                        },
                    );
                    suite.add_result(blob_result);
                }
            }
        }

        suite
    }

    /// Benchmark CSV loading performance
    pub fn benchmark_csv_loading<P: AsRef<Path>>(&self, csvpath: P) -> BenchmarkSuite {
        let mut suite = BenchmarkSuite::new("CSV Loading".to_string());
        let path = csvpath.as_ref();

        if !path.exists() {
            let mut result = BenchmarkResult::new("csv_loading".to_string(), HashMap::new());
            result = result.failure("CSV file not found".to_string());
            suite.add_result(result);
            return suite;
        }

        // Standard CSV loading
        let std_params = HashMap::from([
            ("method".to_string(), "standard".to_string()),
            ("file".to_string(), path.to_string_lossy().to_string()),
        ]);
        let std_result = self.run_benchmark("csv_standard", std_params, || {
            let config = CsvConfig::default().with_header(true);
            match load_csv(path, config) {
                Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
                Err(e) => Err(format!("Failed to load CSV: {e}")),
            }
        });
        suite.add_result(std_result);

        // Parallel CSV loading
        let par_params = HashMap::from([
            ("method".to_string(), "parallel".to_string()),
            ("file".to_string(), path.to_string_lossy().to_string()),
        ]);
        let par_result = self.run_benchmark("csv_parallel", par_params, || {
            let csv_config = CsvConfig::default().with_header(true);
            let streaming_config = StreamingConfig::default()
                .with_parallel(true)
                .with_chunk_size(1000);
            match load_csv_parallel(path, csv_config, streaming_config) {
                Ok(dataset) => Ok((dataset.n_samples(), dataset.n_features())),
                Err(e) => Err(format!("Failed to load CSV in parallel: {e}")),
            }
        });
        suite.add_result(par_result);

        suite
    }

    /// Run comprehensive benchmarks comparing SciRS2 performance
    pub fn run_comprehensive_benchmarks(&self) -> Vec<BenchmarkSuite> {
        println!("Running comprehensive SciRS2 performance benchmarks...\n");

        let mut suites = Vec::new();

        // Toy datasets benchmark
        println!("Benchmarking toy datasets...");
        let toy_suite = self.benchmark_toy_datasets();
        toy_suite.print_summary();
        suites.push(toy_suite);
        println!();

        // Data generation benchmark
        println!("Benchmarking data generation...");
        let gen_suite = self.benchmark_data_generation();
        gen_suite.print_summary();
        suites.push(gen_suite);
        println!();

        suites
    }
}

/// Performance comparison utilities
pub struct PerformanceComparison {
    /// Reference (baseline) results
    pub baseline: BenchmarkSuite,
    /// Current implementation results
    pub current: BenchmarkSuite,
}

impl PerformanceComparison {
    /// Create a new performance comparison
    pub fn new(baseline: BenchmarkSuite, current: BenchmarkSuite) -> Self {
        Self { baseline, current }
    }

    /// Calculate speedup ratio for matching operations
    pub fn calculate_speedups(&self) -> HashMap<String, f64> {
        let mut speedups = HashMap::new();

        for current_result in &self.current.results {
            if let Some(baseline_result) = self
                .baseline
                .results
                .iter()
                .find(|r| r.operation == current_result.operation)
            {
                if baseline_result.success && current_result.success {
                    let speedup = baseline_result.duration.as_secs_f64()
                        / current_result.duration.as_secs_f64();
                    speedups.insert(current_result.operation.clone(), speedup);
                }
            }
        }

        speedups
    }

    /// Print comparison report
    pub fn print_comparison(&self) {
        println!("=== Performance Comparison ===");
        println!("Baseline: {}", self.baseline.name);
        println!("Current:  {}", self.current.name);
        println!();

        let speedups = self.calculate_speedups();

        if speedups.is_empty() {
            println!("No matching operations found for comparison.");
            return;
        }

        let mut improvements = 0;
        let mut regressions = 0;
        let mut total_speedup = 0.0;

        println!("Speedup Analysis:");
        for (operation, speedup) in &speedups {
            let status = if *speedup > 1.1 {
                improvements += 1;
                "🚀 FASTER"
            } else if *speedup < 0.9 {
                regressions += 1;
                "🐌 SLOWER"
            } else {
                "≈ SAME"
            };

            println!("  {operation}: {speedup:.2}x {status}");
            total_speedup += speedup;
        }

        let avg_speedup = total_speedup / speedups.len() as f64;

        println!();
        println!("Summary:");
        println!("  Improvements: {improvements}");
        println!("  Regressions:  {regressions}");
        println!(
            "  Unchanged:    {}",
            speedups.len() - improvements - regressions
        );
        println!("  Average speedup: {avg_speedup:.2}x");

        if avg_speedup > 1.1 {
            println!("  Overall assessment: 🎉 SIGNIFICANT IMPROVEMENT");
        } else if avg_speedup > 1.0 {
            println!("  Overall assessment: ✅ MINOR IMPROVEMENT");
        } else if avg_speedup > 0.9 {
            println!("  Overall assessment: ≈ COMPARABLE PERFORMANCE");
        } else {
            println!("  Overall assessment: ⚠️ PERFORMANCE REGRESSION");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result() {
        let params = HashMap::from([("test".to_string(), "value".to_string())]);
        let result = BenchmarkResult::new("test_op".to_string(), params).success(
            Duration::from_millis(100),
            1000,
            10,
        );

        assert!(result.success);
        assert_eq!(result.samples, 1000);
        assert_eq!(result.features, 10);
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new("test_suite".to_string());

        let result1 = BenchmarkResult::new("op1".to_string(), HashMap::new()).success(
            Duration::from_millis(50),
            500,
            5,
        );
        let result2 = BenchmarkResult::new("op2".to_string(), HashMap::new())
            .failure("test error".to_string());

        suite.add_result(result1);
        suite.add_result(result2);

        assert_eq!(suite.results.len(), 2);
        assert_eq!(suite.successful_results().len(), 1);
        assert_eq!(suite.failed_results().len(), 1);
        assert_eq!(suite.total_samples(), 500);
    }

    #[test]
    fn test_benchmark_runner() {
        let runner = BenchmarkRunner::new().with_iterations(3).with_warmup(1);

        let params = HashMap::new();
        let result = runner.run_benchmark("test", params, || {
            std::thread::sleep(Duration::from_millis(1));
            Ok((100, 10))
        });

        assert!(result.success);
        assert_eq!(result.samples, 100);
        assert_eq!(result.features, 10);
    }

    #[test]
    fn test_toy_datasets_benchmark() {
        let runner = BenchmarkRunner::new().with_iterations(1);
        let suite = runner.benchmark_toy_datasets();

        assert!(!suite.results.is_empty());
        assert!(!suite.successful_results().is_empty());
    }

    #[test]
    fn test_data_generation_benchmark() {
        let runner = BenchmarkRunner::new().with_iterations(1);
        let suite = runner.benchmark_data_generation();

        assert!(!suite.results.is_empty());
        // Allow some failures due to parameter combinations
        assert!(!suite.successful_results().is_empty());
    }
}
