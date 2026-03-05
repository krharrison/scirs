//! Precompiled Metal Shading Language (MSL) kernel source strings.
//!
//! These constants contain MSL source code for common GPU operations used by
//! the SciRS2 Metal backend. Each kernel is a complete, standalone MSL compute
//! kernel that can be compiled at runtime by the Metal API.
//!
//! # Usage
//!
//! ```ignore
//! use scirs2_core::gpu::backends::msl_kernels;
//! let gemm_src: &str = msl_kernels::GEMM_F32;
//! ```

/// General Matrix Multiply (GEMM) kernel for f32.
///
/// Computes C = alpha * A * B + beta * C where A is (M x K), B is (K x N),
/// and C is (M x N). Uses thread-group shared memory for tile-based blocking.
pub const GEMM_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_f32(
    constant float* A      [[buffer(0)]],
    constant float* B      [[buffer(1)]],
    device   float* C      [[buffer(2)]],
    constant uint&  M      [[buffer(3)]],
    constant uint&  K      [[buffer(4)]],
    constant uint&  N      [[buffer(5)]],
    constant float& alpha  [[buffer(6)]],
    constant float& beta   [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}
"#;

/// ReLU activation kernel for f32.
///
/// Applies the ReLU activation function element-wise: y = max(0, x).
pub const RELU_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void relu_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant uint& length      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= length) return;
    output[gid] = max(0.0f, input[gid]);
}
"#;

/// Sigmoid activation kernel for f32.
///
/// Applies the sigmoid activation: y = 1 / (1 + exp(-x)).
pub const SIGMOID_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant uint& length      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= length) return;
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}
"#;

/// Hyperbolic tangent (TanH) activation kernel for f32.
///
/// Applies the tanh activation: y = tanh(x).
pub const TANH_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void tanh_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant uint& length      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= length) return;
    output[gid] = tanh(input[gid]);
}
"#;

/// Gaussian Error Linear Unit (GELU) activation kernel for f32.
///
/// Approximates GELU with: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
/// This is the approximation used in BERT and GPT-style models.
pub const GELU_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gelu_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant uint& length      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= length) return;
    float x = input[gid];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float c = 0.7978845608f;  // sqrt(2 / pi)
    float inner = c * (x + 0.044715f * x * x * x);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}
"#;

/// Parallel sum reduction kernel for f32.
///
/// Reduces an array to its sum using a tree-based reduction in shared memory.
/// The output is a single f32 value written to `output[0]`.
pub const SUM_REDUCTION_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sum_reduction_f32(
    device const float*  input        [[buffer(0)]],
    device       float*  output       [[buffer(1)]],
    constant uint&       length       [[buffer(2)]],
    threadgroup  float*  shared_data  [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint blockDim [[threads_per_threadgroup]])
{
    uint i = bid * blockDim + tid;
    shared_data[tid] = (i < length) ? input[i] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = blockDim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        output[bid] = shared_data[0];
    }
}
"#;

/// Parallel mean reduction kernel for f32.
///
/// Computes the arithmetic mean of an array. Uses a two-pass approach:
/// first reduce to partial sums, then divide by the array length.
pub const MEAN_REDUCTION_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void mean_reduction_f32(
    device const float*  input        [[buffer(0)]],
    device       float*  output       [[buffer(1)]],
    constant uint&       length       [[buffer(2)]],
    threadgroup  float*  shared_data  [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint blockDim [[threads_per_threadgroup]])
{
    uint i = bid * blockDim + tid;
    shared_data[tid] = (i < length) ? input[i] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = blockDim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        output[bid] = shared_data[0] / float(length);
    }
}
"#;

/// Element-wise addition kernel for f32.
pub const ADD_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* c      [[buffer(2)]],
    constant uint&      length [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= length) return;
    c[gid] = a[gid] + b[gid];
}
"#;

/// Element-wise multiplication kernel for f32.
pub const MUL_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void mul_f32(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* c      [[buffer(2)]],
    constant uint&      length [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= length) return;
    c[gid] = a[gid] * b[gid];
}
"#;

/// Softmax kernel for f32.
///
/// Computes the numerically stable softmax over a vector:
/// y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
pub const SOFTMAX_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant uint&      length [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= length) return;

    // Numerically stable: find max first
    float max_val = input[0];
    for (uint i = 1; i < length; ++i) {
        max_val = max(max_val, input[i]);
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (uint i = 0; i < length; ++i) {
        sum += exp(input[i] - max_val);
    }

    output[gid] = exp(input[gid] - max_val) / sum;
}
"#;

/// Layer normalization kernel for f32.
///
/// Normalizes input over the last dimension:
/// y = (x - mean) / sqrt(var + eps) * gamma + beta
pub const LAYER_NORM_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_f32(
    device const float* input   [[buffer(0)]],
    device const float* gamma   [[buffer(1)]],
    device const float* beta    [[buffer(2)]],
    device       float* output  [[buffer(3)]],
    constant uint&      length  [[buffer(4)]],
    constant float&     epsilon [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= length) return;

    // Compute mean
    float mean = 0.0f;
    for (uint i = 0; i < length; ++i) {
        mean += input[i];
    }
    mean /= float(length);

    // Compute variance
    float variance = 0.0f;
    for (uint i = 0; i < length; ++i) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= float(length);

    // Normalize
    output[gid] = gamma[gid] * (input[gid] - mean) / sqrt(variance + epsilon) + beta[gid];
}
"#;
