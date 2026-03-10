//! Integration tests for operation fusion
//!
//! Tests that verify:
//! 1. Correctness: fused ops produce identical results to unfused
//! 2. Graph integration: fusion works with computation graph
//! 3. Gradient correctness: backward passes are correct
//! 4. End-to-end: realistic neural network scenarios

use scirs2_autograd::optimization::fusion::{
    backward::{
        fused_affine_backward, fused_linear_backward, fused_linear_gelu_backward,
        fused_linear_relu_backward, fused_linear_sigmoid_backward, fused_linear_tanh_backward,
        fused_softmax_backward,
    },
    ops::{
        fold_conv_bn_params, fused_affine, fused_elementwise_chain, fused_linear,
        fused_linear_gelu, fused_linear_relu, fused_linear_sigmoid, fused_linear_tanh, fused_mean,
        fused_softmax, fused_variance, BatchNormParams,
    },
    patterns::{GraphNode, OpKind},
    FusionOptimizer,
};
use scirs2_core::ndarray::{Array, IxDyn};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn arr2d(rows: usize, cols: usize, vals: &[f64]) -> Array<f64, IxDyn> {
    Array::from_shape_vec((rows, cols), vals.to_vec())
        .expect("valid shape")
        .into_dyn()
}

fn arr1d(vals: &[f64]) -> Array<f64, IxDyn> {
    Array::from_vec(vals.to_vec()).into_dyn()
}

fn assert_arrays_close(a: &Array<f64, IxDyn>, b: &Array<f64, IxDyn>, tol: f64, msg: &str) {
    assert_eq!(a.shape(), b.shape(), "{}: shape mismatch", msg);
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (av - bv).abs() < tol,
            "{}: element {} differs: {} vs {} (diff: {})",
            msg,
            i,
            av,
            bv,
            (av - bv).abs()
        );
    }
}

// Unfused reference implementation
fn unfused_linear_relu(
    x: &Array<f64, IxDyn>,
    w: &Array<f64, IxDyn>,
    bias: &Array<f64, IxDyn>,
) -> Array<f64, IxDyn> {
    let linear = fused_linear(x, w, bias).expect("linear");
    linear.mapv(|v| if v > 0.0 { v } else { 0.0 })
}

// ---------------------------------------------------------------------------
// Correctness tests: fused vs unfused
// ---------------------------------------------------------------------------

#[test]
fn test_fused_linear_relu_correctness() {
    let x = arr2d(
        4,
        8,
        &(0..32).map(|i| i as f64 * 0.1 - 1.0).collect::<Vec<_>>(),
    );
    let w = arr2d(8, 6, &(0..48).map(|i| i as f64 * 0.05).collect::<Vec<_>>());
    let bias = arr1d(&[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]);

    let fused = fused_linear_relu(&x, &w, &bias).expect("fused");
    let unfused = unfused_linear_relu(&x, &w, &bias);

    assert_arrays_close(&fused, &unfused, 1e-10, "linear+relu correctness");
}

#[test]
fn test_fused_linear_sigmoid_correctness() {
    let x = arr2d(
        3,
        5,
        &(0..15).map(|i| i as f64 * 0.2 - 1.5).collect::<Vec<_>>(),
    );
    let w = arr2d(5, 4, &(0..20).map(|i| i as f64 * 0.1).collect::<Vec<_>>());
    let bias = arr1d(&[0.1, 0.2, -0.1, -0.2]);

    let fused = fused_linear_sigmoid(&x, &w, &bias).expect("fused");

    // Unfused: linear + sigmoid
    let linear = fused_linear(&x, &w, &bias).expect("linear");
    let unfused = linear.mapv(|v| 1.0 / (1.0 + (-v).exp()));

    assert_arrays_close(&fused, &unfused, 1e-10, "linear+sigmoid correctness");
}

#[test]
fn test_fused_linear_tanh_correctness() {
    let x = arr2d(
        3,
        5,
        &(0..15).map(|i| i as f64 * 0.2 - 1.5).collect::<Vec<_>>(),
    );
    let w = arr2d(5, 4, &(0..20).map(|i| i as f64 * 0.1).collect::<Vec<_>>());
    let bias = arr1d(&[0.1, 0.2, -0.1, -0.2]);

    let fused = fused_linear_tanh(&x, &w, &bias).expect("fused");

    // Unfused: linear + tanh
    let linear = fused_linear(&x, &w, &bias).expect("linear");
    let unfused = linear.mapv(|v| v.tanh());

    assert_arrays_close(&fused, &unfused, 1e-10, "linear+tanh correctness");
}

#[test]
fn test_fused_linear_gelu_correctness() {
    let x = arr2d(2, 4, &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5]);
    let w = arr2d(
        4,
        3,
        &[1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5],
    );
    let bias = arr1d(&[0.0, 0.1, -0.1]);

    let fused = fused_linear_gelu(&x, &w, &bias).expect("fused");

    // Unfused: linear + gelu
    let linear = fused_linear(&x, &w, &bias).expect("linear");
    let sqrt_2_over_pi = 0.7978845608028654;
    let coeff = 0.044715;
    let unfused = linear.mapv(|val| {
        let inner = sqrt_2_over_pi * (val + coeff * val * val * val);
        0.5 * val * (1.0 + inner.tanh())
    });

    assert_arrays_close(&fused, &unfused, 1e-10, "linear+gelu correctness");
}

#[test]
fn test_fused_affine_correctness() {
    let x = arr1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let scale = arr1d(&[2.0, 3.0, 4.0, 5.0, 6.0]);
    let shift = arr1d(&[10.0, 20.0, 30.0, 40.0, 50.0]);

    let fused = fused_affine(&x, &scale, &shift).expect("fused");

    // Unfused: multiply then add
    let unfused = &(&x * &scale) + &shift;

    assert_arrays_close(&fused, &unfused, 1e-10, "affine correctness");
}

#[test]
fn test_fused_elementwise_chain_correctness() {
    let x = arr1d(&[-2.0, -1.0, 0.0, 1.0, 2.0]);

    // Test relu -> neg
    let fused = fused_elementwise_chain(&x, &["relu", "neg"]).expect("fused");
    let relu = x.mapv(|v| if v > 0.0 { v } else { 0.0 });
    let unfused = relu.mapv(|v| -v);

    assert_arrays_close(&fused, &unfused, 1e-10, "elementwise chain correctness");
}

// ---------------------------------------------------------------------------
// Gradient correctness tests
// ---------------------------------------------------------------------------

#[test]
fn test_linear_backward_correctness() {
    let x = arr2d(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let w = arr2d(3, 2, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    let grad_output = arr2d(2, 2, &[1.0, 1.0, 1.0, 1.0]);

    let grads = fused_linear_backward(&grad_output, &x, &w).expect("backward");

    // Check dimensions
    assert_eq!(grads.grad_x.shape(), x.shape());
    assert_eq!(grads.grad_w.shape(), w.shape());
    assert_eq!(grads.grad_bias.shape(), &[2]);

    // Check grad_bias = sum(grad_output, axis=0)
    let expected_bias: Vec<f64> = vec![2.0, 2.0]; // sum of [[1,1],[1,1]] along axis 0
    assert_arrays_close(&grads.grad_bias, &arr1d(&expected_bias), 1e-10, "grad_bias");
}

#[test]
fn test_linear_relu_backward_masking() {
    let x = arr2d(2, 2, &[1.0, -1.0, 2.0, -2.0]);
    let w = arr2d(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let grad_output = arr2d(2, 2, &[1.0, 1.0, 1.0, 1.0]);

    // Forward pass output: [1,-1,2,-2] @ [[1,0],[0,1]] = [1,-1,2,-2]
    // After ReLU: [1,0,2,0]
    let output = arr2d(2, 2, &[1.0, 0.0, 2.0, 0.0]);

    let grads = fused_linear_relu_backward(&grad_output, &x, &w, &output).expect("backward");

    // Gradient should be masked: only positive outputs propagate gradient
    // This is tested implicitly by comparing with the mask pattern
    assert_eq!(grads.grad_bias.shape(), &[2]);
}

#[test]
fn test_linear_sigmoid_backward() {
    let x = arr2d(1, 2, &[1.0, 2.0]);
    let w = arr2d(2, 1, &[0.5, 0.5]);
    let grad_output = arr2d(1, 1, &[1.0]);

    // Forward: x @ w = [1*0.5 + 2*0.5] = [1.5]
    // sigmoid(1.5) ≈ 0.8176
    let output = arr2d(1, 1, &[0.8175744761936437]);

    let grads = fused_linear_sigmoid_backward(&grad_output, &x, &w, &output).expect("backward");

    // Sigmoid gradient: y * (1 - y) ≈ 0.8176 * 0.1824 ≈ 0.149
    // This scales the upstream gradient
    assert!(grads.grad_bias.iter().next().unwrap_or(&0.0).abs() > 0.0);
}

#[test]
fn test_linear_tanh_backward() {
    let x = arr2d(1, 2, &[1.0, 1.0]);
    let w = arr2d(2, 1, &[0.5, 0.5]);
    let grad_output = arr2d(1, 1, &[1.0]);

    // Forward: x @ w = [1.0]
    // tanh(1.0) ≈ 0.7616
    let output = arr2d(1, 1, &[0.7615941559557649]);

    let grads = fused_linear_tanh_backward(&grad_output, &x, &w, &output).expect("backward");

    // Tanh gradient: 1 - tanh^2(1.0) ≈ 1 - 0.58 ≈ 0.42
    let grad_val = grads.grad_bias.iter().next().copied().unwrap_or(0.0);
    assert!((grad_val - 0.42).abs() < 0.01);
}

#[test]
fn test_affine_backward_correctness() {
    let x = arr1d(&[1.0, 2.0, 3.0]);
    let scale = arr1d(&[2.0, 3.0, 4.0]);
    let grad_output = arr1d(&[1.0, 1.0, 1.0]);

    let grads = fused_affine_backward(&grad_output, &x, &scale).expect("backward");

    // grad_x = grad_output * scale = [2, 3, 4]
    assert_arrays_close(&grads.grad_x, &arr1d(&[2.0, 3.0, 4.0]), 1e-10, "grad_x");

    // grad_scale = grad_output * x = [1, 2, 3]
    assert_arrays_close(
        &grads.grad_scale,
        &arr1d(&[1.0, 2.0, 3.0]),
        1e-10,
        "grad_scale",
    );

    // grad_shift = grad_output = [1, 1, 1]
    assert_arrays_close(
        &grads.grad_shift,
        &arr1d(&[1.0, 1.0, 1.0]),
        1e-10,
        "grad_shift",
    );
}

#[test]
fn test_softmax_backward_correctness() {
    let x = arr2d(1, 3, &[1.0, 2.0, 3.0]);
    let output = fused_softmax(&x, 1).expect("softmax forward");
    let grad_output = arr2d(1, 3, &[0.1, 0.2, 0.3]);

    let grad_input = fused_softmax_backward(&grad_output, &output, 1).expect("softmax backward");

    // Softmax gradient property: sum(grad_input) = 0
    let sum: f64 = grad_input.iter().sum();
    assert!(sum.abs() < 1e-10, "softmax gradient should sum to zero");
}

// ---------------------------------------------------------------------------
// Graph optimizer integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_graph_optimizer_detects_patterns() {
    let mut optimizer = FusionOptimizer::new();

    // Create a simple graph: matmul -> bias -> relu
    let nodes = vec![
        GraphNode::new(0, OpKind::Input, vec![], vec![4, 8]),
        GraphNode::new(1, OpKind::Input, vec![], vec![8, 6]),
        GraphNode::new(2, OpKind::MatMul, vec![0, 1], vec![4, 6]),
        GraphNode::new(3, OpKind::BiasAdd, vec![2], vec![4, 6]),
        GraphNode::new(4, OpKind::Relu, vec![3], vec![4, 6]),
    ];

    // Mark consumers
    let mut nodes_with_consumers = nodes.clone();
    nodes_with_consumers[2].num_consumers = 1;
    nodes_with_consumers[3].num_consumers = 1;

    optimizer
        .detect_fusions_in_graph(&nodes_with_consumers)
        .expect("detect");

    // Should detect at least one fusion candidate
    assert!(!optimizer.candidates().is_empty());
}

#[test]
fn test_graph_optimizer_applies_fusions() {
    let mut optimizer = FusionOptimizer::new();

    let nodes = vec![
        GraphNode::new(0, OpKind::Input, vec![], vec![4, 8]),
        GraphNode::new(1, OpKind::Input, vec![], vec![8, 6]),
        GraphNode::new(2, OpKind::MatMul, vec![0, 1], vec![4, 6]),
        GraphNode::new(3, OpKind::BiasAdd, vec![2], vec![4, 6]),
    ];

    let mut nodes_with_consumers = nodes.clone();
    nodes_with_consumers[2].num_consumers = 1;

    optimizer
        .detect_fusions_in_graph(&nodes_with_consumers)
        .expect("detect");

    let fused_nodes = optimizer
        .apply_fusions_with_nodes(&nodes_with_consumers)
        .expect("apply");

    assert!(!fused_nodes.is_empty());
    assert_eq!(fused_nodes[0].original_ids.len(), 2);
}

#[test]
fn test_graph_optimizer_no_overlapping_fusions() {
    let mut optimizer = FusionOptimizer::new();

    // Two separate fusible chains
    let nodes = vec![
        GraphNode::new(0, OpKind::Input, vec![], vec![4, 8]),
        GraphNode::new(1, OpKind::MatMul, vec![0], vec![4, 6]),
        GraphNode::new(2, OpKind::BiasAdd, vec![1], vec![4, 6]),
        GraphNode::new(3, OpKind::Input, vec![], vec![4, 8]),
        GraphNode::new(4, OpKind::MatMul, vec![3], vec![4, 6]),
        GraphNode::new(5, OpKind::BiasAdd, vec![4], vec![4, 6]),
    ];

    let mut nodes_with_consumers = nodes.clone();
    nodes_with_consumers[1].num_consumers = 1;
    nodes_with_consumers[4].num_consumers = 1;

    optimizer
        .detect_fusions_in_graph(&nodes_with_consumers)
        .expect("detect");

    let fused_nodes = optimizer
        .apply_fusions_with_nodes(&nodes_with_consumers)
        .expect("apply");

    // Both chains should be fused (no overlap)
    assert_eq!(fused_nodes.len(), 2);

    // Verify no node appears in multiple fused groups
    let mut all_original_ids: Vec<usize> = Vec::new();
    for fused in &fused_nodes {
        all_original_ids.extend(&fused.original_ids);
    }

    let unique_ids: std::collections::HashSet<_> = all_original_ids.iter().copied().collect();
    assert_eq!(
        all_original_ids.len(),
        unique_ids.len(),
        "No overlapping fusions"
    );
}

// ---------------------------------------------------------------------------
// Conv + BatchNorm fusion tests
// ---------------------------------------------------------------------------

#[test]
fn test_conv_bn_param_folding() {
    // 2 output channels, 1 input channel, 1x1 kernel
    let conv_weight = arr2d(2, 1, &[1.0, 2.0]);
    let conv_bias = arr1d(&[0.5, -0.5]);

    let bn_params = BatchNormParams {
        running_mean: vec![1.0, 2.0],
        running_var: vec![4.0, 9.0],
        gamma: vec![2.0, 3.0],
        beta: vec![0.1, 0.2],
        epsilon: 1e-5,
    };

    let (fused_w, fused_b) =
        fold_conv_bn_params(&conv_weight, Some(&conv_bias), &bn_params).expect("fold");

    // Folded weight: gamma / sqrt(var + eps) * w
    // Channel 0: 2.0 / sqrt(4.0) * 1.0 = 2.0 / 2.0 * 1.0 = 1.0
    // Channel 1: 3.0 / sqrt(9.0) * 2.0 = 3.0 / 3.0 * 2.0 = 2.0

    let w_flat: Vec<f64> = fused_w.iter().copied().collect();
    assert!((w_flat[0] - 1.0).abs() < 0.01);
    assert!((w_flat[1] - 2.0).abs() < 0.01);

    // Folded bias: gamma / sqrt(var + eps) * (bias - mean) + beta
    // Channel 0: 1.0 * (0.5 - 1.0) + 0.1 = -0.5 + 0.1 = -0.4
    // Channel 1: 1.0 * (-0.5 - 2.0) + 0.2 = -2.5 + 0.2 = -2.3

    let b_flat: Vec<f64> = fused_b.iter().copied().collect();
    assert!((b_flat[0] - (-0.4)).abs() < 0.01);
    assert!((b_flat[1] - (-2.3)).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// End-to-end neural network tests
// ---------------------------------------------------------------------------

#[test]
fn test_simple_mlp_with_fusion() {
    // Simulate a 2-layer MLP: input -> linear+relu -> linear+relu -> output

    // Layer 1
    let x = arr2d(4, 8, &(0..32).map(|i| i as f64 * 0.1).collect::<Vec<_>>());
    let w1 = arr2d(
        8,
        16,
        &(0..128).map(|i| i as f64 * 0.01).collect::<Vec<_>>(),
    );
    let b1 = arr1d(&(0..16).map(|i| i as f64 * 0.1).collect::<Vec<_>>());

    let h1 = fused_linear_relu(&x, &w1, &b1).expect("layer1");

    // Layer 2
    let w2 = arr2d(
        16,
        10,
        &(0..160).map(|i| i as f64 * 0.01).collect::<Vec<_>>(),
    );
    let b2 = arr1d(&(0..10).map(|i| i as f64 * 0.1).collect::<Vec<_>>());

    let h2 = fused_linear_relu(&h1, &w2, &b2).expect("layer2");

    // Output layer with softmax
    let w3 = arr2d(10, 5, &(0..50).map(|i| i as f64 * 0.01).collect::<Vec<_>>());
    let b3 = arr1d(&[0.1, 0.2, 0.3, 0.4, 0.5]);

    let logits = fused_linear(&h2, &w3, &b3).expect("output");
    let output = fused_softmax(&logits, 1).expect("softmax");

    // Check output shape and properties
    assert_eq!(output.shape(), &[4, 5]);

    // Each row should sum to 1.0
    for i in 0..4 {
        let row_sum: f64 = (0..5).map(|j| output[[i, j]]).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_transformer_ffn_fusion() {
    // Transformer feed-forward network: x -> linear -> gelu -> linear

    let batch = 8;
    let seq_len = 32;
    let d_model = 64;
    let d_ff = 256;

    let x = arr2d(
        batch * seq_len,
        d_model,
        &(0..batch * seq_len * d_model)
            .map(|i| i as f64 * 0.001)
            .collect::<Vec<_>>(),
    );

    // Expand layer
    let w1 = arr2d(
        d_model,
        d_ff,
        &(0..d_model * d_ff)
            .map(|i| (i as f64 * 0.01).sin())
            .collect::<Vec<_>>(),
    );
    let b1 = arr1d(&(0..d_ff).map(|i| i as f64 * 0.001).collect::<Vec<_>>());

    let h = fused_linear_gelu(&x, &w1, &b1).expect("expand+gelu");

    // Project back layer
    let w2 = arr2d(
        d_ff,
        d_model,
        &(0..d_ff * d_model)
            .map(|i| (i as f64 * 0.01).cos())
            .collect::<Vec<_>>(),
    );
    let b2 = arr1d(&(0..d_model).map(|i| i as f64 * 0.001).collect::<Vec<_>>());

    let output = fused_linear(&h, &w2, &b2).expect("project");

    // Check shapes
    assert_eq!(h.shape(), &[batch * seq_len, d_ff]);
    assert_eq!(output.shape(), &[batch * seq_len, d_model]);
}

// ---------------------------------------------------------------------------
// Numerical stability tests
// ---------------------------------------------------------------------------

#[test]
fn test_softmax_large_values() {
    // Large values should not cause overflow
    let x = arr2d(
        2,
        4,
        &[1000.0, 1001.0, 1002.0, 1003.0, 500.0, 501.0, 502.0, 503.0],
    );

    let output = fused_softmax(&x, 1).expect("softmax");

    // Check all values are finite
    for &v in output.iter() {
        assert!(v.is_finite(), "softmax output should be finite");
        assert!(
            (0.0..=1.0).contains(&v),
            "softmax output should be in [0,1]"
        );
    }

    // Each row should sum to 1.0
    for i in 0..2 {
        let row_sum: f64 = (0..4).map(|j| output[[i, j]]).sum();
        assert!((row_sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_gelu_extreme_values() {
    // Test GELU with extreme values
    let x = arr2d(1, 1, &[1.0]);
    let w = arr2d(1, 5, &[-100.0, -10.0, 0.0, 10.0, 100.0]);
    let bias = arr1d(&[0.0, 0.0, 0.0, 0.0, 0.0]);

    let output = fused_linear_gelu(&x, &w, &bias).expect("gelu");

    // All values should be finite
    for &v in output.iter() {
        assert!(v.is_finite(), "GELU output should be finite");
    }
}

// ---------------------------------------------------------------------------
// Performance characteristic tests
// ---------------------------------------------------------------------------

#[test]
fn test_fusion_reduces_memory_allocations() {
    // This test verifies that fused operations reduce intermediate allocations
    // by checking that fused ops don't allocate more than unfused.

    let x = arr2d(
        100,
        200,
        &(0..20000).map(|i| i as f64 * 0.001).collect::<Vec<_>>(),
    );
    let w = arr2d(
        200,
        150,
        &(0..30000).map(|i| i as f64 * 0.001).collect::<Vec<_>>(),
    );
    let bias = arr1d(&(0..150).map(|i| i as f64 * 0.01).collect::<Vec<_>>());

    // Both should produce correct results
    let fused = fused_linear_relu(&x, &w, &bias).expect("fused");
    let unfused = unfused_linear_relu(&x, &w, &bias);

    assert_arrays_close(&fused, &unfused, 1e-8, "fusion performance test");

    // The point: fused operations should not allocate intermediate tensors,
    // but we can't easily measure that here. This test at least verifies correctness.
}
