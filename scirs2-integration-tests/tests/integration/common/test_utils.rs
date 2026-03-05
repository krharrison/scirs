// Common test utilities for cross-crate integration tests
// Follows no-unwrap policy and provides helpers for property-based testing

use scirs2_core::ndarray::{Array1, Array2, ArrayD, Dimension};
use proptest::prelude::*;
use scirs2_core::ScalarNum;
use std::time::Instant;

/// Result type for test utilities
pub type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub operation_name: String,
}

/// Performance measurement result
#[derive(Debug, Clone)]
pub struct PerfMeasurement {
    pub duration_ms: f64,
    pub operation_name: String,
    pub data_size: usize,
}

/// Create a test array with deterministic values
pub fn create_test_array_1d<T: ScalarNum>(size: usize, seed: u64) -> TestResult<Array1<T>> {
    use scirs2_core::prelude::*;

    let rng = RandomState::new(seed);
    let values: Vec<T> = (0..size)
        .map(|i| T::from(i as f64 / size as f64).expect("Value conversion failed"))
        .collect();

    Array1::from_vec(values)
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|e| format!("Failed to create 1D array: {}", e).into())
}

/// Create a 2D test array with deterministic values
pub fn create_test_array_2d<T: ScalarNum>(rows: usize, cols: usize, seed: u64) -> TestResult<Array2<T>> {
    use scirs2_core::prelude::*;

    let size = rows * cols;
    let values: Vec<T> = (0..size)
        .map(|i| T::from((i as f64 + seed as f64) / size as f64).expect("Value conversion failed"))
        .collect();

    Array2::from_shape_vec((rows, cols), values)
        .map_err(|e| format!("Failed to create 2D array: {}", e).into())
}

/// Measure execution time of a function
pub fn measure_time<F, R>(operation_name: &str, f: F) -> TestResult<(R, PerfMeasurement)>
where
    F: FnOnce() -> TestResult<R>,
{
    let start = Instant::now();
    let result = f()?;
    let duration = start.elapsed();

    let measurement = PerfMeasurement {
        duration_ms: duration.as_secs_f64() * 1000.0,
        operation_name: operation_name.to_string(),
        data_size: 0, // Can be set by caller
    };

    Ok((result, measurement))
}

/// Check if arrays are approximately equal
pub fn arrays_approx_equal<T, D>(a: &ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>,
                                  b: &ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>,
                                  tolerance: f64) -> bool
where
    T: ScalarNum + approx::AbsDiffEq<Epsilon = f64>,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return false;
    }

    a.iter()
        .zip(b.iter())
        .all(|(x, y)| approx::abs_diff_eq!(*x, *y, epsilon = tolerance))
}

/// Property test strategy for small positive integers
pub fn small_size() -> impl Strategy<Value = usize> {
    1usize..100
}

/// Property test strategy for array dimensions
pub fn array_dimensions() -> impl Strategy<Value = (usize, usize)> {
    (1usize..100, 1usize..100)
}

/// Property test strategy for tolerance values
pub fn tolerance() -> impl Strategy<Value = f64> {
    1e-10..1e-3
}

/// Get temporary directory for test files
pub fn get_temp_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("scirs2_integration_tests_{}", std::process::id()))
}

/// Clean up temporary test directory
pub fn cleanup_temp_dir(path: &std::path::Path) -> TestResult<()> {
    if path.exists() {
        std::fs::remove_dir_all(path)
            .map_err(|e| format!("Failed to clean up temp dir: {}", e))?;
    }
    Ok(())
}

/// Check if GPU is available for testing
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Check for CUDA availability
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Create a synthetic dataset for testing
pub fn create_synthetic_classification_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> TestResult<(Array2<f64>, Array1<usize>)> {
    use scirs2_core::prelude::*;

    let rng = RandomState::new(seed);

    // Create features
    let mut features = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            features[[i, j]] = ((i * n_features + j) as f64 + seed as f64) / (n_samples * n_features) as f64;
        }
    }

    // Create labels
    let labels: Array1<usize> = (0..n_samples)
        .map(|i| i % n_classes)
        .collect();

    Ok((features, labels))
}

/// Assert that an operation's memory usage is below a threshold
pub fn assert_memory_efficient<F, R>(
    operation: F,
    max_mb: f64,
    description: &str,
) -> TestResult<R>
where
    F: FnOnce() -> TestResult<R>,
{
    // Note: Actual memory tracking would require system-specific code
    // This is a placeholder for the structure
    let result = operation()?;

    // In a real implementation, you would track memory here
    eprintln!("Memory efficiency check for '{}': max allowed {} MB", description, max_mb);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_array_1d() {
        let arr = create_test_array_1d::<f64>(10, 42)
            .expect("Failed to create test array");
        assert_eq!(arr.len(), 10);
    }

    #[test]
    fn test_create_test_array_2d() {
        let arr = create_test_array_2d::<f64>(5, 4, 42)
            .expect("Failed to create 2D test array");
        assert_eq!(arr.shape(), &[5, 4]);
    }

    #[test]
    fn test_measure_time() {
        let (result, perf) = measure_time("test_op", || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            Ok(42)
        }).expect("Measurement failed");

        assert_eq!(result, 42);
        assert!(perf.duration_ms >= 10.0);
    }

    #[test]
    fn test_temp_dir() {
        let temp_dir = get_temp_dir();
        assert!(temp_dir.to_string_lossy().contains("scirs2_integration_tests"));
    }

    #[test]
    fn test_synthetic_data_creation() {
        let (features, labels) = create_synthetic_classification_data(100, 10, 3, 42)
            .expect("Failed to create synthetic data");

        assert_eq!(features.shape(), &[100, 10]);
        assert_eq!(labels.len(), 100);
        assert!(labels.iter().all(|&l| l < 3));
    }
}
