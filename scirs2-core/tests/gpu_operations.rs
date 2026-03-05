#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
//! Integration tests for GPU operations

use scirs2_core::gpu::{GpuBackend, GpuContext};

#[test]
fn test_gpu_gemm_works() {
    let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
    let a = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    let b = ctx.create_buffer_from_slice(&[5.0f32, 6.0, 7.0, 8.0]);
    let c = ctx.gemm(&a, &b, 2, 2, 2).expect("GEMM failed");
    assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_gpu_relu_works() {
    let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
    let input = ctx.create_buffer_from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let output = ctx.relu(&input).expect("ReLU failed");
    assert_eq!(output.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_gpu_sigmoid_works() {
    let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
    let input = ctx.create_buffer_from_slice(&[0.0f32]);
    let output = ctx.sigmoid(&input).expect("Sigmoid failed");
    let result = output.to_vec();
    assert!((result[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_gpu_tanh_works() {
    let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
    let input = ctx.create_buffer_from_slice(&[0.0f32]);
    let output = ctx.tanh(&input).expect("Tanh failed");
    let result = output.to_vec();
    assert!((result[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_gpu_gelu_works() {
    let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
    let input = ctx.create_buffer_from_slice(&[0.0f32]);
    let output = ctx.gelu(&input).expect("GELU failed");
    let result = output.to_vec();
    assert!((result[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_gpu_sum_all_works() {
    let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
    let input = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let output = ctx.sum_all(&input).expect("Sum all failed");
    let result = output.to_vec();
    assert_eq!(result.len(), 1);
    assert!((result[0] - 15.0).abs() < 1e-6);
}

#[test]
fn test_gpu_mean_all_works() {
    let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
    let input = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let output = ctx.mean_all(&input).expect("Mean all failed");
    let result = output.to_vec();
    assert_eq!(result.len(), 1);
    assert!((result[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_gpu_broadcast_works() {
    let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
    let input = ctx.create_buffer_from_slice(&[1.0f32, 2.0]);
    let output = ctx
        .broadcast(&input, &[2], &[3, 2])
        .expect("Broadcast failed");
    assert_eq!(output.to_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
}

#[test]
fn test_gpu_scale_works() {
    let ctx = GpuContext::new(GpuBackend::Cpu).expect("Failed to create context");
    let input = ctx.create_buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    let output = ctx.scale(&input, 2.0).expect("Scale failed");
    assert_eq!(output.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
}
