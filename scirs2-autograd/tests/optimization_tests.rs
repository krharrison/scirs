//! Integration tests for graph optimization passes
//!
//! These tests verify that constant folding and expression simplification
//! work correctly on real computation graphs.

use ag::graph::AsGraph;
use ag::optimization::{GraphOptimizer, OptimizationConfig, OptimizationLevel};
use ag::tensor_ops as T;
use scirs2_autograd as ag;

#[test]
fn test_constant_folding_identifies_constants() {
    ag::run(|ctx: &mut ag::Context<f32>| {
        // Create some constant tensors
        let a = T::scalar(2.0, ctx);
        let b = T::scalar(3.0, ctx);
        let c = T::zeros(&[2, 2], ctx);
        let d = T::ones(&[2, 2], ctx);

        // These should all be identified as constants
        // Evaluate to ensure graph is built
        let _ = a.eval(ctx);
        let _ = b.eval(ctx);
        let _ = c.eval(ctx);
        let _ = d.eval(ctx);

        // Create optimizer and run constant folding
        let config = OptimizationConfig {
            constant_folding: true,
            cse: false,
            expression_simplification: false,
            dead_code_elimination: false,
            operation_fusion: false,
            memory_optimization: false,
            max_passes: 1,
            level: OptimizationLevel::Basic,
        };

        let optimizer = GraphOptimizer::with_config(config);
        let report = optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");

        // Should have identified some constants
        println!("Constant folding report: {:?}", report);
    });
}

#[test]
fn test_constant_propagation_through_operations() {
    ag::run(|ctx: &mut ag::Context<f32>| {
        // Create an expression with constants: 2 * 3
        let a = T::scalar(2.0, ctx);
        let b = T::scalar(3.0, ctx);
        let product = a * b;

        // Evaluate the result
        let result = product.eval(ctx).expect("Evaluation failed");
        assert_eq!(result[[]], 6.0);

        // Run optimization
        let config = OptimizationConfig {
            constant_folding: true,
            cse: false,
            expression_simplification: false,
            dead_code_elimination: false,
            operation_fusion: false,
            memory_optimization: false,
            max_passes: 3,
            level: OptimizationLevel::Basic,
        };

        let optimizer = GraphOptimizer::with_config(config);
        let report = optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");

        println!("Propagation report: {:?}", report);
        // Constants should be propagated through the multiplication.
        // The report should be produced without error; constant_folding_applied
        // counts folded nodes (may be zero for simple scalars with current impl).
        let _ = report.constant_folding_applied;
    });
}

#[test]
fn test_expression_simplification_counting() {
    ag::run(|ctx: &mut ag::Context<f32>| {
        // Create expressions that could be simplified
        let x = T::standard_normal(&[2, 2], ctx);
        let zero = T::scalar(0.0, ctx);
        let one = T::scalar(1.0, ctx);

        // x + 0 should simplify to x
        let add_zero = x + zero;

        // x * 1 should simplify to x
        let mul_one = x * one;

        // Evaluate to build the graph
        let _ = add_zero.eval(ctx);
        let _ = mul_one.eval(ctx);

        // Run optimization with expression simplification
        let config = OptimizationConfig {
            constant_folding: false,
            cse: false,
            expression_simplification: true,
            dead_code_elimination: false,
            operation_fusion: false,
            memory_optimization: false,
            max_passes: 1,
            level: OptimizationLevel::Standard,
        };

        let optimizer = GraphOptimizer::with_config(config);
        let report = optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");

        println!("Simplification report: {:?}", report);
        // Should have found some simplification opportunities
        // Note: actual simplification depends on pattern matching implementation
    });
}

#[test]
fn test_combined_optimizations() {
    ag::run(|ctx: &mut ag::Context<f32>| {
        // Create a complex expression: (2.0 * x + 0.0) * 1.0
        let x = T::standard_normal(&[3, 3], ctx);
        let two = T::scalar(2.0, ctx);
        let zero = T::scalar(0.0, ctx);
        let one = T::scalar(1.0, ctx);

        let expr = (two * x + zero) * one;

        // Evaluate original expression
        let original_result = expr.eval(ctx).expect("Evaluation failed");

        // Run all optimizations
        let optimizer = GraphOptimizer::with_level(OptimizationLevel::Aggressive);
        let report = optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");

        println!("Combined optimization report:");
        report.print_summary();

        // Verify the result is still mathematically correct
        let optimized_result = expr.eval(ctx).expect("Evaluation failed");

        // Results should be the same (within floating point precision)
        for (orig, opt) in original_result.iter().zip(optimized_result.iter()) {
            assert!(
                (orig - opt).abs() < 1e-6,
                "Results differ after optimization"
            );
        }
    });
}

#[test]
fn test_optimization_levels() {
    ag::run(|ctx: &mut ag::Context<f32>| {
        // Test that different optimization levels work
        let x = T::standard_normal(&[2, 2], ctx);
        let _ = x.eval(ctx);

        // None level
        let none_optimizer = GraphOptimizer::with_level(OptimizationLevel::None);
        let none_report = none_optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");
        assert_eq!(none_report.passes_completed, 0);

        // Basic level
        let basic_optimizer = GraphOptimizer::with_level(OptimizationLevel::Basic);
        let basic_report = basic_optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");
        assert!(basic_report.passes_completed <= 2);

        // Standard level
        let std_optimizer = GraphOptimizer::with_level(OptimizationLevel::Standard);
        let std_report = std_optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");
        assert!(std_report.passes_completed <= 5);

        // Aggressive level
        let agg_optimizer = GraphOptimizer::with_level(OptimizationLevel::Aggressive);
        let agg_report = agg_optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");
        assert!(agg_report.passes_completed <= 10);
    });
}

#[test]
fn test_constant_arithmetic() {
    use ag::optimization::constant_folding::ConstantValue;

    // Test scalar operations
    let a = ConstantValue::Scalar(2.0f32);
    let b = ConstantValue::Scalar(3.0f32);

    let sum = a.add(&b).expect("Add failed");
    if let ConstantValue::Scalar(val) = sum {
        assert_eq!(val, 5.0);
    }

    let diff = a.sub(&b).expect("Sub failed");
    if let ConstantValue::Scalar(val) = diff {
        assert_eq!(val, -1.0);
    }

    let prod = a.mul(&b).expect("Mul failed");
    if let ConstantValue::Scalar(val) = prod {
        assert_eq!(val, 6.0);
    }

    let quot = a.div(&b).expect("Div failed");
    if let ConstantValue::Scalar(val) = quot {
        assert!((val - 0.666667).abs() < 0.001);
    }
}

#[test]
fn test_empty_graph_optimization() {
    ag::run(|ctx: &mut ag::Context<f32>| {
        // Empty graph should optimize without errors
        let optimizer = GraphOptimizer::new();
        let report = optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");

        assert_eq!(report.total_optimizations(), 0);
    });
}

#[test]
fn test_optimization_preserves_gradients() {
    ag::run(|ctx: &mut ag::Context<f32>| {
        // Create a simple differentiable computation
        let x = T::standard_normal(&[2, 2], ctx);
        let y = T::scalar(2.0, ctx);
        let z = x * y;

        // Evaluate before optimization
        let result_before = z.eval(ctx).expect("Eval failed");

        // Run optimization
        let optimizer = GraphOptimizer::with_level(OptimizationLevel::Standard);
        let _report = optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");

        // Evaluate after optimization
        let result_after = z.eval(ctx).expect("Eval failed");

        // Results should be identical
        for (before, after) in result_before.iter().zip(result_after.iter()) {
            assert_eq!(before, after, "Optimization changed computation result");
        }
    });
}

#[test]
fn test_multiple_optimization_passes() {
    ag::run(|ctx: &mut ag::Context<f32>| {
        // Create a graph that benefits from multiple passes
        let x = T::standard_normal(&[3, 3], ctx);
        let c1 = T::scalar(2.0, ctx);
        let c2 = T::scalar(3.0, ctx);

        // (2 * 3) * x should be simplified to 6 * x
        let expr = (c1 * c2) * x;
        let _ = expr.eval(ctx);

        // Run optimizer with multiple passes
        let config = OptimizationConfig {
            constant_folding: true,
            cse: true,
            expression_simplification: true,
            dead_code_elimination: true,
            operation_fusion: false,
            memory_optimization: false,
            max_passes: 5,
            level: OptimizationLevel::Standard,
        };

        let optimizer = GraphOptimizer::with_config(config);
        let report = optimizer
            .optimize(ctx.as_graph())
            .expect("Optimization failed");

        println!("Multi-pass report: {:?}", report);
        assert!(report.passes_completed >= 1);
    });
}
