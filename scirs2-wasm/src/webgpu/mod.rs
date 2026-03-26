//! WebGPU backend for accelerated matrix operations.
//!
//! Provides GPU-accelerated (or CPU-fallback) computation with WGSL shader generation
//! for use in browser environments with WebGPU support.
//!
//! # Architecture
//!
//! | Module | Role |
//! |--------|------|
//! | `types` | Core types: `WebGpuConfig`, `GpuBuffer`, `GpuError`, `GpuBufferDescriptor`, etc. |
//! | `shader_gen` | WGSL source generation: matmul, elementwise, reduction, conv1d |
//! | `matmul` | Tiled GEMM with CPU simulation and WGSL shader output |
//! | `operations` | Elementwise and reduction operations (CPU + WGSL) |
//! | `backend` | `DeviceSimulator` and `WebGpuContext` — full CPU-side execution model |
//! | `wasm_bindings` | `WasmWebGpu` and `#[wasm_bindgen]` entry points |
//!
//! # Quick start (native Rust)
//!
//! ```rust
//! use scirs2_wasm::webgpu::backend::WebGpuContext;
//! use scirs2_wasm::webgpu::shader_gen::ElementwiseOp;
//! use scirs2_wasm::webgpu::types::{GpuBufferUsage, WebGpuConfig};
//!
//! let mut ctx = WebGpuContext::new(WebGpuConfig::default());
//!
//! // Upload input data.
//! let a_id = ctx.upload_buffer(vec![1.0_f32, 2.0, 3.0, 4.0], GpuBufferUsage::Storage).unwrap();
//! let b_id = ctx.upload_buffer(vec![5.0_f32, 6.0, 7.0, 8.0], GpuBufferUsage::Storage).unwrap();
//!
//! // Matrix multiply 2×2 × 2×2.
//! let c_id = ctx.matmul(a_id, b_id, 2, 2, 2).unwrap();
//! let c = ctx.download_buffer(c_id).unwrap();
//! assert!((c[0] - 19.0).abs() < 1e-4);
//! ```

pub mod backend;
pub mod matmul;
pub mod operations;
pub mod shader_gen;
pub mod types;
pub mod wasm_bindings;

pub use backend::{BufferId, DeviceSimulator, WebGpuContext};
pub use matmul::WebGpuMatmul;
pub use operations::WebGpuOps;
pub use shader_gen::{
    generate_conv1d_shader, generate_elementwise_shader, generate_matmul_shader,
    generate_reduction_shader, ElementwiseOp, ReductionOp, WgslGenerator,
};
pub use types::{
    ComputePipelineDescriptor, GpuBuffer, GpuBufferDescriptor, GpuBufferUsage, GpuError,
    WebGpuBackend, WebGpuConfig, WebGpuResult,
};
pub use wasm_bindings::{
    matmul_f32, reduce_max_f32, reduce_sum_f32, relu_f32, sigmoid_f32, WasmWebGpu,
};
