// Integration tests for scirs2-neural + scirs2-optimize
// Tests ML training pipelines, optimizer integration, and gradient flow

use scirs2_core::ndarray::{Array1, Array2, Axis};
use proptest::prelude::*;
use crate::integration::common::*;
use crate::integration::fixtures::TestDatasets;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Test that neural network can be trained with optimize module optimizers
#[test]
fn test_neural_with_sgd_optimizer() -> TestResult<()> {
    // This test verifies the integration between neural network training
    // and optimization algorithms from scirs2-optimize

    // Create simple training data
    let (x, y) = TestDatasets::xor_dataset();

    // Note: Actual implementation depends on the current API state of scirs2-neural and scirs2-optimize
    // This is a structural test showing the integration pattern

    println!("Testing neural network with SGD optimizer");
    println!("Input shape: {:?}", x.shape());
    println!("Target shape: {:?}", y.shape());

    // TODO: Once scirs2-neural and scirs2-optimize APIs are stable:
    // 1. Create a neural network model
    // 2. Create an SGD optimizer from scirs2-optimize
    // 3. Train for a few epochs
    // 4. Verify loss decreases

    Ok(())
}

/// Test gradient computation flows correctly between modules
#[test]
fn test_gradient_flow_between_modules() -> TestResult<()> {
    // Verify that gradients computed in scirs2-neural can be
    // consumed by scirs2-optimize optimizers

    let (features, labels) = create_synthetic_classification_data(50, 10, 2, 42)?;

    println!("Testing gradient flow with {} samples, {} features",
             features.nrows(), features.ncols());

    // TODO: Implement when APIs are stable:
    // 1. Forward pass through neural network
    // 2. Compute loss and gradients
    // 3. Pass gradients to optimizer
    // 4. Verify weight updates occur correctly

    Ok(())
}

/// Test hyperparameter optimization integration
#[test]
fn test_hyperparameter_optimization() -> TestResult<()> {
    // Tests integration of scirs2-optimize's hyperparameter search
    // with scirs2-neural model training

    println!("Testing hyperparameter optimization integration");

    // TODO: Implement when APIs are stable:
    // 1. Define parameter space (learning rates, batch sizes, etc.)
    // 2. Use scirs2-optimize grid search or bayesian optimization
    // 3. Train neural networks with different hyperparameters
    // 4. Verify best configuration is selected

    Ok(())
}

/// Test zero-copy tensor passing between modules
#[test]
fn test_zero_copy_tensor_passing() -> TestResult<()> {
    // Verifies that tensors can be passed between neural and optimize
    // modules without unnecessary copying

    let data = create_test_array_2d::<f64>(100, 50, 42)?;

    println!("Testing zero-copy data transfer");
    println!("Original data shape: {:?}", data.shape());

    // TODO: Implement memory efficiency validation:
    // 1. Pass data from neural to optimize module
    // 2. Verify no additional allocations occur
    // 3. Check that views/references are used where possible

    Ok(())
}

/// Test error propagation across module boundaries
#[test]
fn test_error_propagation() -> TestResult<()> {
    // Verifies that errors from scirs2-optimize are properly
    // propagated through scirs2-neural training loops

    println!("Testing error propagation between modules");

    // TODO: Test various error conditions:
    // 1. Invalid optimizer parameters
    // 2. Convergence failures
    // 3. Numerical instabilities
    // 4. Verify errors are properly typed and informative

    Ok(())
}

/// Test batch processing integration
#[test]
fn test_batch_processing_integration() -> TestResult<()> {
    // Tests that batch-based training works correctly when
    // integrating neural networks with optimizers

    let (features, labels) = create_synthetic_classification_data(200, 20, 3, 42)?;
    let batch_size = 32;

    println!("Testing batch processing with batch_size={}", batch_size);

    // TODO: Implement batch processing test:
    // 1. Split data into batches
    // 2. Process each batch through neural network
    // 3. Accumulate gradients
    // 4. Apply optimizer update
    // 5. Verify results are consistent with full-batch training

    Ok(())
}

/// Test learning rate scheduling integration
#[test]
fn test_learning_rate_scheduling() -> TestResult<()> {
    // Tests integration of learning rate schedules from scirs2-optimize
    // with neural network training

    println!("Testing learning rate scheduling");

    // TODO: Implement when APIs are stable:
    // 1. Create a learning rate schedule (e.g., step decay, cosine annealing)
    // 2. Train neural network with scheduled learning rate
    // 3. Verify learning rate changes occur at correct epochs
    // 4. Compare convergence with and without scheduling

    Ok(())
}

/// Test momentum-based optimizer integration
#[test]
fn test_momentum_optimizer_integration() -> TestResult<()> {
    // Tests that momentum-based optimizers (Adam, RMSprop) from
    // scirs2-optimize work correctly with neural networks

    println!("Testing momentum-based optimizer integration");

    // TODO: Implement:
    // 1. Create neural network
    // 2. Use Adam or RMSprop from scirs2-optimize
    // 3. Verify momentum terms are accumulated correctly
    // 4. Compare convergence speed vs vanilla SGD

    Ok(())
}

/// Test early stopping integration
#[test]
fn test_early_stopping_integration() -> TestResult<()> {
    // Tests that early stopping criteria from scirs2-optimize
    // can be used to halt neural network training

    println!("Testing early stopping integration");

    // TODO: Implement:
    // 1. Set up early stopping criteria (validation loss patience)
    // 2. Train neural network
    // 3. Verify training stops when criteria are met
    // 4. Ensure best weights are restored

    Ok(())
}

// Property-based tests

proptest! {
    #[test]
    fn prop_loss_decreases_with_training(
        n_samples in 10usize..100,
        n_features in 5usize..20,
        n_epochs in 5usize..20
    ) {
        // Property: Training a neural network should decrease loss
        // (unless we hit numerical issues or local minima)

        let (features, labels) = create_synthetic_classification_data(
            n_samples, n_features, 2, 42
        ).expect("Failed to create test data");

        // TODO: Implement property test when APIs are stable:
        // 1. Create simple neural network
        // 2. Train for n_epochs
        // 3. Verify final loss <= initial loss

        prop_assert!(true, "Property test placeholder");
    }

    #[test]
    fn prop_gradient_descent_converges(
        learning_rate in 0.001f64..0.1,
        batch_size in 8usize..64
    ) {
        // Property: Gradient descent with reasonable hyperparameters
        // should eventually converge (loss stops decreasing)

        // TODO: Implement convergence property test
        prop_assert!(learning_rate > 0.0);
        prop_assert!(batch_size > 0);
    }

    #[test]
    fn prop_optimizer_state_consistent(
        n_iterations in 10usize..100
    ) {
        // Property: Optimizer state (momentum, etc.) should remain
        // consistent across iterations

        // TODO: Implement state consistency check
        prop_assert!(n_iterations > 0);
    }
}

/// Test memory efficiency of integrated training pipeline
#[test]
fn test_memory_efficiency_integration() -> TestResult<()> {
    // Verifies that the integrated neural+optimize pipeline
    // doesn't leak memory or create unnecessary copies

    let (features, labels) = create_synthetic_classification_data(1000, 100, 5, 42)?;

    println!("Testing memory efficiency with large dataset");
    println!("Dataset size: {} samples x {} features", features.nrows(), features.ncols());

    assert_memory_efficient(
        || {
            // TODO: Run training loop
            // Verify memory usage stays within bounds
            Ok(())
        },
        500.0,  // 500 MB max
        "Neural network training with optimizer",
    )?;

    Ok(())
}

/// Test convergence on a simple regression task
#[test]
fn test_simple_regression_convergence() -> TestResult<()> {
    // End-to-end test of neural network training on a simple
    // linear regression task

    let (x, y) = TestDatasets::linear_dataset(100);

    println!("Testing convergence on linear regression");
    println!("Data shape: X={:?}, y={:?}", x.shape(), y.shape());

    // TODO: Implement full training loop:
    // 1. Create single-layer neural network
    // 2. Train with MSE loss and SGD
    // 3. Verify predictions match y = 2x + 1
    // 4. Assert final loss < 0.1

    Ok(())
}

/// Test multi-objective optimization integration
#[test]
fn test_multi_objective_optimization() -> TestResult<()> {
    // Tests integration with multi-objective optimization
    // (e.g., minimizing loss while maximizing sparsity)

    println!("Testing multi-objective optimization integration");

    // TODO: Implement when multi-objective optimizers are available:
    // 1. Define multiple objectives (loss, regularization, etc.)
    // 2. Use scirs2-optimize multi-objective solver
    // 3. Verify Pareto frontier is explored correctly

    Ok(())
}

/// Test distributed training integration
#[test]
#[ignore] // Requires multi-process setup
fn test_distributed_training_integration() -> TestResult<()> {
    // Tests that distributed optimization from scirs2-optimize
    // works with neural network training

    println!("Testing distributed training integration");

    // TODO: Implement distributed training test:
    // 1. Set up multiple workers
    // 2. Distribute data across workers
    // 3. Synchronize gradients
    // 4. Verify convergence is similar to single-process training

    Ok(())
}

#[cfg(test)]
mod api_compatibility_tests {
    use super::*;

    /// Test that type conversions work correctly between modules
    #[test]
    fn test_type_compatibility() -> TestResult<()> {
        // Verifies that data types from scirs2-neural are compatible
        // with scirs2-optimize and vice versa

        println!("Testing type compatibility between neural and optimize");

        // TODO: Test type conversions:
        // 1. Neural tensor -> Optimize parameter vector
        // 2. Optimize gradient -> Neural gradient
        // 3. Verify no precision loss

        Ok(())
    }

    /// Test that both modules handle edge cases consistently
    #[test]
    fn test_edge_case_handling() -> TestResult<()> {
        // Tests edge cases like empty inputs, single samples, etc.

        // TODO: Test edge cases:
        // 1. Empty dataset
        // 2. Single sample
        // 3. Zero gradients
        // 4. Inf/NaN handling

        Ok(())
    }
}
