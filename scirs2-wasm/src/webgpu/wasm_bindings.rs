//! WebAssembly bindings for the WebGPU backend.
//!
//! Provides `wasm-bindgen`-annotated entry points that expose GPU-accelerated
//! (or CPU-fallback) operations to JavaScript/TypeScript.
//!
//! On non-WASM targets the same functions are available as plain Rust
//! free functions, making it easy to unit-test them on any host platform.

use crate::webgpu::backend::WebGpuContext;
use crate::webgpu::shader_gen::{ElementwiseOp, ReductionOp};
use crate::webgpu::types::{GpuBufferUsage, GpuError, WebGpuConfig, WebGpuResult};

// ============================================================
// WasmWebGpu struct
// ============================================================

/// Browser-facing WebGPU compute context.
///
/// Wraps a `WebGpuContext` and exposes operations as `#[wasm_bindgen]`
/// methods so they can be called from JavaScript.
///
/// # JavaScript example
/// ```js
/// import init, { WasmWebGpu } from 'scirs2-wasm';
/// await init();
///
/// const gpu = new WasmWebGpu();
/// const a = new Float32Array([1, 2, 3, 4]);
/// const b = new Float32Array([5, 6, 7, 8]);
/// const c = gpu.js_matmul(a, b, 2, 2, 2);
/// // c ≈ [19, 22, 43, 50]
/// ```
pub struct WasmWebGpu {
    ctx: WebGpuContext,
}

impl WasmWebGpu {
    /// Create a new `WasmWebGpu` with default configuration.
    pub fn new() -> Self {
        Self {
            ctx: WebGpuContext::new(WebGpuConfig::default()),
        }
    }

    /// Create a `WasmWebGpu` from a custom `WebGpuConfig`.
    pub fn with_config(config: WebGpuConfig) -> Self {
        Self {
            ctx: WebGpuContext::new(config),
        }
    }

    // ------------------------------------------------------------------
    // Matrix multiply
    // ------------------------------------------------------------------

    /// Compute `C = A × B` where A is (m × k) and B is (k × n).
    ///
    /// Returns a flat `Vec<f32>` of length `m * n` in row-major order, or an
    /// error string on failure.
    ///
    /// # Arguments
    /// * `a` — flat `f32` slice of length `m * k`, row-major.
    /// * `b` — flat `f32` slice of length `k * n`, row-major.
    /// * `m`, `n`, `k` — matrix dimensions.
    pub fn js_matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: u32,
        n: u32,
        k: u32,
    ) -> WebGpuResult<Vec<f32>> {
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let a_id = self
            .ctx
            .upload_buffer(a.to_vec(), GpuBufferUsage::Storage)?;
        let b_id = self
            .ctx
            .upload_buffer(b.to_vec(), GpuBufferUsage::Storage)?;
        let c_id = self.ctx.matmul(a_id, b_id, m, k, n)?;
        self.ctx.download_buffer(c_id)
    }

    // ------------------------------------------------------------------
    // Elementwise operations
    // ------------------------------------------------------------------

    /// Apply ReLU element-wise: `out[i] = max(0, data[i])`.
    pub fn js_elementwise_relu(&mut self, data: &[f32]) -> WebGpuResult<Vec<f32>> {
        let id = self
            .ctx
            .upload_buffer(data.to_vec(), GpuBufferUsage::Storage)?;
        let out_id = self.ctx.elementwise(id, None, ElementwiseOp::Relu)?;
        self.ctx.download_buffer(out_id)
    }

    /// Apply sigmoid element-wise: `out[i] = 1 / (1 + exp(-data[i]))`.
    pub fn js_elementwise_sigmoid(&mut self, data: &[f32]) -> WebGpuResult<Vec<f32>> {
        let id = self
            .ctx
            .upload_buffer(data.to_vec(), GpuBufferUsage::Storage)?;
        let out_id = self.ctx.elementwise(id, None, ElementwiseOp::Sigmoid)?;
        self.ctx.download_buffer(out_id)
    }

    /// Apply exp element-wise: `out[i] = exp(data[i])`.
    pub fn js_elementwise_exp(&mut self, data: &[f32]) -> WebGpuResult<Vec<f32>> {
        let id = self
            .ctx
            .upload_buffer(data.to_vec(), GpuBufferUsage::Storage)?;
        let out_id = self.ctx.elementwise(id, None, ElementwiseOp::Exp)?;
        self.ctx.download_buffer(out_id)
    }

    /// Apply log element-wise: `out[i] = ln(data[i])`.
    pub fn js_elementwise_log(&mut self, data: &[f32]) -> WebGpuResult<Vec<f32>> {
        let id = self
            .ctx
            .upload_buffer(data.to_vec(), GpuBufferUsage::Storage)?;
        let out_id = self.ctx.elementwise(id, None, ElementwiseOp::Log)?;
        self.ctx.download_buffer(out_id)
    }

    /// Element-wise addition: `out[i] = a[i] + b[i]`.
    pub fn js_elementwise_add(&mut self, a: &[f32], b: &[f32]) -> WebGpuResult<Vec<f32>> {
        if a.len() != b.len() {
            return Err(GpuError::Execution(format!(
                "add: length mismatch {} vs {}",
                a.len(),
                b.len()
            )));
        }
        let a_id = self
            .ctx
            .upload_buffer(a.to_vec(), GpuBufferUsage::Storage)?;
        let b_id = self
            .ctx
            .upload_buffer(b.to_vec(), GpuBufferUsage::Storage)?;
        let out_id = self.ctx.elementwise(a_id, Some(b_id), ElementwiseOp::Add)?;
        self.ctx.download_buffer(out_id)
    }

    // ------------------------------------------------------------------
    // Reduction operations
    // ------------------------------------------------------------------

    /// Compute the sum of all elements in `data`.
    pub fn js_reduction_sum(&mut self, data: &[f32]) -> WebGpuResult<f32> {
        let id = self
            .ctx
            .upload_buffer(data.to_vec(), GpuBufferUsage::Storage)?;
        self.ctx.reduce(id, ReductionOp::Sum)
    }

    /// Compute the maximum element in `data`.
    pub fn js_reduction_max(&mut self, data: &[f32]) -> WebGpuResult<f32> {
        let id = self
            .ctx
            .upload_buffer(data.to_vec(), GpuBufferUsage::Storage)?;
        self.ctx.reduce(id, ReductionOp::Max)
    }

    /// Compute the minimum element in `data`.
    pub fn js_reduction_min(&mut self, data: &[f32]) -> WebGpuResult<f32> {
        let id = self
            .ctx
            .upload_buffer(data.to_vec(), GpuBufferUsage::Storage)?;
        self.ctx.reduce(id, ReductionOp::Min)
    }
}

impl Default for WasmWebGpu {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Non-wasm free functions (also serve as the non-wasm fallback path)
// ============================================================

/// Compute `C = A × B` without allocating a `WasmWebGpu`.
///
/// This function is always available and works identically on native and WASM.
pub fn matmul_f32(a: &[f32], b: &[f32], m: u32, n: u32, k: u32) -> WebGpuResult<Vec<f32>> {
    WasmWebGpu::new().js_matmul(a, b, m, n, k)
}

/// Apply ReLU without allocating a persistent context.
pub fn relu_f32(data: &[f32]) -> WebGpuResult<Vec<f32>> {
    WasmWebGpu::new().js_elementwise_relu(data)
}

/// Apply sigmoid without allocating a persistent context.
pub fn sigmoid_f32(data: &[f32]) -> WebGpuResult<Vec<f32>> {
    WasmWebGpu::new().js_elementwise_sigmoid(data)
}

/// Sum all elements without allocating a persistent context.
pub fn reduce_sum_f32(data: &[f32]) -> WebGpuResult<f32> {
    WasmWebGpu::new().js_reduction_sum(data)
}

/// Find the maximum without allocating a persistent context.
pub fn reduce_max_f32(data: &[f32]) -> WebGpuResult<f32> {
    WasmWebGpu::new().js_reduction_max(data)
}

// ============================================================
// wasm-bindgen interface (WASM target only)
// ============================================================
//
// The functions below are identical to the free functions above but carry
// `#[wasm_bindgen]` annotations so wasm-pack can export them.
// They are conditionally compiled so that native tests do not need the
// wasm-bindgen runtime.

#[cfg(target_arch = "wasm32")]
mod wasm_export {
    use super::*;
    use wasm_bindgen::prelude::*;

    /// Compute `C = A × B` (WASM entry point).
    ///
    /// Returns `Float32Array` of length `m * n` or throws a JS `Error`.
    #[wasm_bindgen(js_name = "gpu_matmul")]
    pub fn wasm_matmul(a: &[f32], b: &[f32], m: u32, n: u32, k: u32) -> Result<Vec<f32>, JsValue> {
        matmul_f32(a, b, m, n, k).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Apply ReLU (WASM entry point).
    #[wasm_bindgen(js_name = "gpu_relu")]
    pub fn wasm_relu(data: &[f32]) -> Result<Vec<f32>, JsValue> {
        relu_f32(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Apply sigmoid (WASM entry point).
    #[wasm_bindgen(js_name = "gpu_sigmoid")]
    pub fn wasm_sigmoid(data: &[f32]) -> Result<Vec<f32>, JsValue> {
        sigmoid_f32(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Reduce sum (WASM entry point).
    #[wasm_bindgen(js_name = "gpu_reduce_sum")]
    pub fn wasm_reduce_sum(data: &[f32]) -> Result<f32, JsValue> {
        reduce_sum_f32(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Reduce max (WASM entry point).
    #[wasm_bindgen(js_name = "gpu_reduce_max")]
    pub fn wasm_reduce_max(data: &[f32]) -> Result<f32, JsValue> {
        reduce_max_f32(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// ============================================================
// Tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> WasmWebGpu {
        WasmWebGpu::new()
    }

    // ---- matmul ----

    #[test]
    fn test_js_matmul_2x2() {
        let mut g = gpu();
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0, 8.0];
        let c = g.js_matmul(&a, &b, 2, 2, 2).expect("matmul");
        let expected = [19.0_f32, 22.0, 43.0, 50.0];
        for (r, &e) in c.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4, "got {r}, expected {e}");
        }
    }

    #[test]
    fn test_matmul_f32_free_fn() {
        let a = [1.0_f32, 0.0, 0.0, 1.0]; // identity
        let b = [3.0_f32, 7.0, 2.0, 5.0];
        let c = matmul_f32(&a, &b, 2, 2, 2).expect("matmul");
        // identity × B = B
        for (r, &e) in c.iter().zip(b.iter()) {
            assert!((r - e).abs() < 1e-4, "identity matmul: {r} != {e}");
        }
    }

    // ---- relu ----

    #[test]
    fn test_js_relu_clips_negatives() {
        let mut g = gpu();
        let data = [-3.0_f32, -0.5, 0.0, 1.0, 4.0];
        let out = g.js_elementwise_relu(&data).expect("relu");
        assert_eq!(out, [0.0_f32, 0.0, 0.0, 1.0, 4.0]);
    }

    #[test]
    fn test_relu_f32_free_fn() {
        let out = relu_f32(&[-1.0_f32, 2.0, -3.0]).expect("relu");
        assert_eq!(out, [0.0_f32, 2.0, 0.0]);
    }

    // ---- sigmoid ----

    #[test]
    fn test_js_sigmoid_in_range() {
        let mut g = gpu();
        let data: Vec<f32> = (-10..=10).map(|x| x as f32).collect();
        let out = g.js_elementwise_sigmoid(&data).expect("sigmoid");
        for &v in &out {
            assert!(v > 0.0 && v < 1.0, "sigmoid out of (0,1): {v}");
        }
    }

    #[test]
    fn test_sigmoid_f32_free_fn() {
        let out = sigmoid_f32(&[0.0_f32]).expect("sigmoid");
        // sigmoid(0) = 0.5
        assert!((out[0] - 0.5).abs() < 1e-5, "sigmoid(0) should be 0.5");
    }

    // ---- reduction ----

    #[test]
    fn test_js_reduction_sum_equals_direct() {
        let mut g = gpu();
        let data: Vec<f32> = (1..=50).map(|x| x as f32).collect();
        let expected: f32 = data.iter().sum();
        let sum = g.js_reduction_sum(&data).expect("sum");
        assert!((sum - expected).abs() < 1.0, "sum {sum} != {expected}");
    }

    #[test]
    fn test_js_reduction_max_equals_direct() {
        let mut g = gpu();
        let data = vec![3.0_f32, 1.0, 4.0, 1.5, 9.0, 2.6];
        let sum = g.js_reduction_max(&data).expect("max");
        assert!((sum - 9.0).abs() < 1e-5, "max should be 9.0");
    }

    #[test]
    fn test_reduce_sum_free_fn() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let sum = reduce_sum_f32(&data).expect("sum");
        assert!((sum - 10.0).abs() < 1e-5, "sum should be 10.0");
    }

    #[test]
    fn test_reduce_max_free_fn() {
        let data = vec![5.0_f32, 3.0, 8.0, 1.0];
        let max = reduce_max_f32(&data).expect("max");
        assert!((max - 8.0).abs() < 1e-5, "max should be 8.0");
    }

    // ---- add ----

    #[test]
    fn test_js_elementwise_add() {
        let mut g = gpu();
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let out = g.js_elementwise_add(&a, &b).expect("add");
        assert_eq!(out, [5.0_f32, 7.0, 9.0]);
    }

    #[test]
    fn test_js_elementwise_add_length_mismatch_fails() {
        let mut g = gpu();
        let result = g.js_elementwise_add(&[1.0_f32], &[1.0_f32, 2.0]);
        assert!(result.is_err(), "length mismatch should be an error");
    }

    // ---- exp / log ----

    #[test]
    fn test_js_elementwise_exp() {
        let mut g = gpu();
        let out = g.js_elementwise_exp(&[0.0_f32, 1.0]).expect("exp");
        assert!((out[0] - 1.0).abs() < 1e-5, "exp(0)=1");
        assert!((out[1] - std::f32::consts::E).abs() < 1e-4, "exp(1)=e");
    }

    #[test]
    fn test_js_elementwise_log() {
        let mut g = gpu();
        let out = g
            .js_elementwise_log(&[1.0_f32, std::f32::consts::E])
            .expect("log");
        assert!(out[0].abs() < 1e-5, "log(1)=0");
        assert!((out[1] - 1.0).abs() < 1e-4, "log(e)=1");
    }

    // ---- WasmWebGpu::default ----

    #[test]
    fn test_wasm_webgpu_default() {
        let g = WasmWebGpu::default();
        assert!(!g.ctx.is_gpu_available());
    }

    // ---- with_config ----

    #[test]
    fn test_with_config_custom_tile() {
        let cfg = WebGpuConfig {
            workgroup_size_x: 4,
            ..WebGpuConfig::default()
        };
        let mut g = WasmWebGpu::with_config(cfg);
        // Simple 1×1 × 1×1 matmul just to verify the config is active.
        let out = g
            .js_matmul(&[3.0_f32], &[4.0_f32], 1, 1, 1)
            .expect("matmul");
        assert!((out[0] - 12.0).abs() < 1e-4, "1x1 matmul should be 12");
    }
}
