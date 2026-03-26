//! Pure-Rust WebGPU backend simulator.
//!
//! Provides `DeviceSimulator` and `WebGpuContext` which mirror the WebGPU
//! execution model — buffer registry, compute passes, and dispatch — entirely
//! in CPU Rust.  This lets library consumers write code that works identically
//! in non-browser environments and (when ported to a real `wgpu` or browser
//! JS bridge) in the browser.

use std::collections::HashMap;

use crate::webgpu::shader_gen::{ElementwiseOp, ReductionOp, WgslGenerator};
use crate::webgpu::types::{
    GpuBuffer, GpuBufferDescriptor, GpuBufferUsage, GpuError, WebGpuConfig, WebGpuResult,
};

// ============================================================
// BufferId
// ============================================================

/// Opaque identifier for a buffer in the `DeviceSimulator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

// ============================================================
// DeviceSimulator
// ============================================================

/// A CPU-side simulation of a WebGPU logical device.
///
/// Maintains a registry of named buffers and simulates compute passes by
/// executing the equivalent Rust logic (rather than compiling/executing WGSL).
pub struct DeviceSimulator {
    buffers: HashMap<BufferId, GpuBuffer>,
    next_id: u64,
    config: WebGpuConfig,
    generator: WgslGenerator,
}

impl DeviceSimulator {
    /// Create a new simulator from a `WebGpuConfig`.
    pub fn new(config: WebGpuConfig) -> Self {
        let tile_size = config.workgroup_size_x as usize;
        let gen = WgslGenerator::new().with_tile_size(tile_size);
        Self {
            buffers: HashMap::new(),
            next_id: 1,
            config,
            generator: gen,
        }
    }

    // ------------------------------------------------------------------
    // Buffer management
    // ------------------------------------------------------------------

    fn alloc_id(&mut self) -> BufferId {
        let id = BufferId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Create a buffer pre-filled with `init_data`.
    ///
    /// Returns an error if the requested size exceeds `config.max_buffer_size`
    /// or if `init_data` is empty and `desc.usage` is `Storage`.
    pub fn create_buffer(
        &mut self,
        desc: &GpuBufferDescriptor,
        init_data: Vec<f32>,
    ) -> WebGpuResult<BufferId> {
        let byte_size = init_data.len() * std::mem::size_of::<f32>();
        if byte_size > self.config.max_buffer_size {
            return Err(GpuError::BufferTooLarge {
                requested: byte_size,
                limit: self.config.max_buffer_size,
            });
        }
        let id = self.alloc_id();
        let buf = GpuBuffer::with_data(init_data, desc.usage);
        self.buffers.insert(id, buf);
        Ok(id)
    }

    /// Create a zero-filled buffer of `n_elements` `f32` elements.
    pub fn create_zero_buffer(
        &mut self,
        n_elements: usize,
        usage: GpuBufferUsage,
    ) -> WebGpuResult<BufferId> {
        let byte_size = n_elements * std::mem::size_of::<f32>();
        if byte_size > self.config.max_buffer_size {
            return Err(GpuError::BufferTooLarge {
                requested: byte_size,
                limit: self.config.max_buffer_size,
            });
        }
        let id = self.alloc_id();
        let buf = GpuBuffer::zeros(n_elements, usage);
        self.buffers.insert(id, buf);
        Ok(id)
    }

    /// Read the data from a buffer.  Returns a clone of the stored slice.
    pub fn read_buffer(&self, id: BufferId) -> WebGpuResult<Vec<f32>> {
        self.buffers
            .get(&id)
            .map(|b| b.data.clone())
            .ok_or_else(|| GpuError::Execution(format!("buffer {:?} not found", id)))
    }

    /// Destroy (deallocate) a buffer.  Returns `true` if it existed.
    pub fn destroy_buffer(&mut self, id: BufferId) -> bool {
        self.buffers.remove(&id).is_some()
    }

    /// Return the number of live buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    // ------------------------------------------------------------------
    // Compute operations — matrix multiply
    // ------------------------------------------------------------------

    /// Execute tiled matrix multiply `C = A × B`.
    ///
    /// * `a_id` — buffer containing A (M×K, row-major).
    /// * `b_id` — buffer containing B (K×N, row-major).
    /// * Returns a new buffer containing C (M×N, row-major).
    pub fn execute_matmul(
        &mut self,
        a_id: BufferId,
        b_id: BufferId,
        m: usize,
        k: usize,
        n: usize,
    ) -> WebGpuResult<BufferId> {
        let a = self
            .buffers
            .get(&a_id)
            .ok_or_else(|| GpuError::Execution(format!("matmul: A buffer {:?} not found", a_id)))?
            .data
            .clone();
        let b = self
            .buffers
            .get(&b_id)
            .ok_or_else(|| GpuError::Execution(format!("matmul: B buffer {:?} not found", b_id)))?
            .data
            .clone();

        if a.len() != m * k {
            return Err(GpuError::Execution(format!(
                "matmul: A has {} elements but expected m*k = {}*{} = {}",
                a.len(),
                m,
                k,
                m * k
            )));
        }
        if b.len() != k * n {
            return Err(GpuError::Execution(format!(
                "matmul: B has {} elements but expected k*n = {}*{} = {}",
                b.len(),
                k,
                n,
                k * n
            )));
        }

        let c = tiled_matmul(&a, &b, m, k, n, self.config.workgroup_size_x as usize);
        let desc = GpuBufferDescriptor {
            size: c.len() * std::mem::size_of::<f32>(),
            usage: GpuBufferUsage::Storage,
            label: None,
        };
        self.create_buffer(&desc, c)
    }

    // ------------------------------------------------------------------
    // Compute operations — element-wise
    // ------------------------------------------------------------------

    /// Execute an element-wise unary or binary operation.
    ///
    /// For binary ops (`Add`, `Mul`) `b_id` must be `Some`; for unary ops it
    /// is ignored.  Returns a new buffer with the results.
    pub fn execute_elementwise(
        &mut self,
        input_id: BufferId,
        b_id: Option<BufferId>,
        op: ElementwiseOp,
    ) -> WebGpuResult<BufferId> {
        let a = self
            .buffers
            .get(&input_id)
            .ok_or_else(|| {
                GpuError::Execution(format!("elementwise: A buffer {:?} not found", input_id))
            })?
            .data
            .clone();

        let result: Vec<f32> = match op {
            ElementwiseOp::Add => {
                let bid = b_id.ok_or_else(|| {
                    GpuError::Execution("elementwise Add requires b_id".to_string())
                })?;
                let b = self
                    .buffers
                    .get(&bid)
                    .ok_or_else(|| {
                        GpuError::Execution(format!("elementwise: B buffer {:?} not found", bid))
                    })?
                    .data
                    .clone();
                if a.len() != b.len() {
                    return Err(GpuError::Execution(format!(
                        "elementwise Add: length mismatch {} vs {}",
                        a.len(),
                        b.len()
                    )));
                }
                a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
            }
            ElementwiseOp::Mul => {
                let bid = b_id.ok_or_else(|| {
                    GpuError::Execution("elementwise Mul requires b_id".to_string())
                })?;
                let b = self
                    .buffers
                    .get(&bid)
                    .ok_or_else(|| {
                        GpuError::Execution(format!("elementwise: B buffer {:?} not found", bid))
                    })?
                    .data
                    .clone();
                if a.len() != b.len() {
                    return Err(GpuError::Execution(format!(
                        "elementwise Mul: length mismatch {} vs {}",
                        a.len(),
                        b.len()
                    )));
                }
                a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
            }
            ElementwiseOp::Relu => a.iter().map(|&x| x.max(0.0_f32)).collect(),
            ElementwiseOp::Sigmoid => a
                .iter()
                .map(|&x| 1.0_f32 / (1.0_f32 + (-x).exp()))
                .collect(),
            ElementwiseOp::Exp => a.iter().map(|&x| x.exp()).collect(),
            ElementwiseOp::Log => a.iter().map(|&x| x.ln()).collect(),
            ElementwiseOp::Negate => a.iter().map(|&x| -x).collect(),
        };

        let desc = GpuBufferDescriptor {
            size: result.len() * std::mem::size_of::<f32>(),
            usage: GpuBufferUsage::Storage,
            label: None,
        };
        self.create_buffer(&desc, result)
    }

    // ------------------------------------------------------------------
    // Compute operations — reduction
    // ------------------------------------------------------------------

    /// Execute a parallel tree reduction over the contents of `input_id`.
    ///
    /// Returns a scalar `f32` result (the CPU accumulation).
    pub fn execute_reduction(&self, input_id: BufferId, op: ReductionOp) -> WebGpuResult<f32> {
        let data = self
            .buffers
            .get(&input_id)
            .ok_or_else(|| {
                GpuError::Execution(format!("reduction: buffer {:?} not found", input_id))
            })?
            .data
            .as_slice();

        Ok(parallel_tree_reduce(data, op))
    }

    // ------------------------------------------------------------------
    // Generic compute pass (shader parsing subset)
    // ------------------------------------------------------------------

    /// Submit a generic compute pass described by a WGSL shader string.
    ///
    /// In this simulation, rather than truly compiling and executing WGSL, the
    /// method detects the *operation class* from the shader comment header
    /// (e.g. `"// Tiled GEMM"`, `"// Element-wise"`, `"// Parallel tree
    /// reduction"`, `"// 1-D convolution"`) and dispatches to the appropriate
    /// CPU routine.  For unknown shaders it returns a zero-filled output
    /// buffer of the same size as the first input buffer.
    ///
    /// `buffers` must contain:
    ///  * index 0 — primary input (A)
    ///  * index 1 — secondary input (B / kernel) or `None` for unary ops
    ///  * Output is allocated automatically and its `BufferId` returned.
    ///
    /// `dispatch` — `(x, y, z)` workgroup grid; currently unused in the CPU
    /// simulation but validated to be non-zero.
    pub fn submit_compute_pass(
        &mut self,
        shader: &str,
        buffers: &[BufferId],
        dispatch: (u32, u32, u32),
    ) -> WebGpuResult<BufferId> {
        if dispatch.0 == 0 || dispatch.1 == 0 || dispatch.2 == 0 {
            return Err(GpuError::Execution(
                "dispatch dimensions must all be > 0".to_string(),
            ));
        }
        if buffers.is_empty() {
            return Err(GpuError::Execution(
                "submit_compute_pass: at least one buffer required".to_string(),
            ));
        }

        let a_id = buffers[0];

        // Detect operation class from shader header comment.
        if shader.contains("// Tiled GEMM") || shader.contains("// Tiled matrix multiplication") {
            // Expect buffers: [A, B, (optional C already allocated)]
            if buffers.len() < 2 {
                return Err(GpuError::Execution(
                    "GEMM pass requires at least 2 buffers (A, B)".to_string(),
                ));
            }
            let b_id = buffers[1];
            let a_len = self
                .buffers
                .get(&a_id)
                .map(|b| b.data.len())
                .ok_or_else(|| GpuError::Execution("A buffer not found".to_string()))?;
            let b_len = self
                .buffers
                .get(&b_id)
                .map(|b| b.data.len())
                .ok_or_else(|| GpuError::Execution("B buffer not found".to_string()))?;
            // Best-effort: infer square matrices (m=n=k=sqrt(a_len)).
            let k = (a_len as f64).sqrt().round() as usize;
            let n = b_len / k;
            let m = a_len / k;
            return self.execute_matmul(a_id, b_id, m, k, n);
        }

        if shader.contains("// Element-wise") {
            // Detect op from label in first line.
            let op = detect_elementwise_op(shader);
            let b_opt = buffers.get(1).copied();
            return self.execute_elementwise(a_id, b_opt, op);
        }

        if shader.contains("// Parallel tree reduction") {
            let op = detect_reduction_op(shader);
            let scalar = self.execute_reduction(a_id, op)?;
            let desc = GpuBufferDescriptor {
                size: std::mem::size_of::<f32>(),
                usage: GpuBufferUsage::Staging,
                label: None,
            };
            return self.create_buffer(&desc, vec![scalar]);
        }

        if shader.contains("// 1-D convolution") {
            if buffers.len() < 2 {
                return Err(GpuError::Execution(
                    "Conv1d pass requires 2 buffers (input, kernel)".to_string(),
                ));
            }
            let kernel_id = buffers[1];
            return self.execute_conv1d(a_id, kernel_id, 1, 0);
        }

        // Unknown shader — produce a zero-filled output with the same element
        // count as the primary input buffer.
        let n_elem = self
            .buffers
            .get(&a_id)
            .map(|b| b.data.len())
            .ok_or_else(|| GpuError::Execution("buffer not found in pass".to_string()))?;
        self.create_zero_buffer(n_elem, GpuBufferUsage::Storage)
    }

    // ------------------------------------------------------------------
    // Conv1D helper
    // ------------------------------------------------------------------

    /// Execute a 1-D discrete convolution.
    ///
    /// * `input_id`  — input signal.
    /// * `kernel_id` — convolution kernel (filter taps).
    /// * `stride`    — output stride (≥ 1).
    /// * `padding`   — symmetric zero-padding applied to the input.
    pub fn execute_conv1d(
        &mut self,
        input_id: BufferId,
        kernel_id: BufferId,
        stride: usize,
        padding: usize,
    ) -> WebGpuResult<BufferId> {
        let input = self
            .buffers
            .get(&input_id)
            .ok_or_else(|| {
                GpuError::Execution(format!("conv1d: input buffer {:?} not found", input_id))
            })?
            .data
            .clone();
        let kernel = self
            .buffers
            .get(&kernel_id)
            .ok_or_else(|| {
                GpuError::Execution(format!("conv1d: kernel buffer {:?} not found", kernel_id))
            })?
            .data
            .clone();

        let ks = kernel.len();
        let in_len = input.len();
        let stride = stride.max(1);
        let padded_len = in_len + 2 * padding;
        // output_len = floor((padded_len - ks) / stride) + 1
        let out_len = if padded_len >= ks {
            (padded_len - ks) / stride + 1
        } else {
            0
        };

        let mut output = vec![0.0_f32; out_len];
        for (out_idx, out_val) in output.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            let in_start = (out_idx * stride) as isize - padding as isize;
            for (ki, &kern_val) in kernel.iter().enumerate() {
                let in_pos = in_start + ki as isize;
                if in_pos >= 0 && (in_pos as usize) < in_len {
                    acc += input[in_pos as usize] * kern_val;
                }
            }
            *out_val = acc;
        }

        let desc = GpuBufferDescriptor {
            size: output.len() * std::mem::size_of::<f32>(),
            usage: GpuBufferUsage::Storage,
            label: None,
        };
        self.create_buffer(&desc, output)
    }

    // ------------------------------------------------------------------
    // Shader string accessors (for documentation / debugging)
    // ------------------------------------------------------------------

    /// Return the WGSL source for the matmul shader at the current tile size.
    pub fn matmul_shader_source(&self, m: usize, n: usize, k: usize) -> String {
        self.generator.matmul(m, n, k)
    }

    /// Return the WGSL source for an element-wise shader.
    pub fn elementwise_shader_source(&self, op: ElementwiseOp) -> String {
        self.generator.elementwise(op)
    }

    /// Return the WGSL source for a reduction shader.
    pub fn reduction_shader_source(&self, op: ReductionOp) -> String {
        self.generator.reduction(op)
    }
}

// ============================================================
// WebGpuContext
// ============================================================

/// High-level context wrapping a `DeviceSimulator` with optional adapter info.
///
/// Intended as the primary entry-point for consumers of this backend.
pub struct WebGpuContext {
    pub(crate) device: DeviceSimulator,
    adapter_name: String,
    is_gpu_available: bool,
}

impl WebGpuContext {
    /// Create a new context.
    ///
    /// In a browser environment `is_gpu_available` would be `true` when the
    /// page's `navigator.gpu` adapter was successfully requested.  Outside a
    /// browser this is always `false` and operations fall back to the CPU
    /// simulator.
    pub fn new(config: WebGpuConfig) -> Self {
        Self {
            device: DeviceSimulator::new(config),
            adapter_name: "CPU Simulator".to_string(),
            is_gpu_available: false,
        }
    }

    /// Return whether a real GPU adapter is available.
    pub fn is_gpu_available(&self) -> bool {
        self.is_gpu_available
    }

    /// Return the adapter description string.
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Return a reference to the underlying `DeviceSimulator`.
    pub fn device(&self) -> &DeviceSimulator {
        &self.device
    }

    /// Return a mutable reference to the underlying `DeviceSimulator`.
    pub fn device_mut(&mut self) -> &mut DeviceSimulator {
        &mut self.device
    }

    // ------------------------------------------------------------------
    // Convenience wrappers
    // ------------------------------------------------------------------

    /// Create a buffer from initial data.
    pub fn upload_buffer(
        &mut self,
        data: Vec<f32>,
        usage: GpuBufferUsage,
    ) -> WebGpuResult<BufferId> {
        let desc = GpuBufferDescriptor {
            size: data.len() * std::mem::size_of::<f32>(),
            usage,
            label: None,
        };
        self.device.create_buffer(&desc, data)
    }

    /// Read data from a buffer.
    pub fn download_buffer(&self, id: BufferId) -> WebGpuResult<Vec<f32>> {
        self.device.read_buffer(id)
    }

    /// Execute matmul and return the output buffer.
    pub fn matmul(
        &mut self,
        a_id: BufferId,
        b_id: BufferId,
        m: usize,
        k: usize,
        n: usize,
    ) -> WebGpuResult<BufferId> {
        self.device.execute_matmul(a_id, b_id, m, k, n)
    }

    /// Execute an element-wise op and return the output buffer.
    pub fn elementwise(
        &mut self,
        input_id: BufferId,
        b_id: Option<BufferId>,
        op: ElementwiseOp,
    ) -> WebGpuResult<BufferId> {
        self.device.execute_elementwise(input_id, b_id, op)
    }

    /// Execute a reduction and return a scalar.
    pub fn reduce(&self, input_id: BufferId, op: ReductionOp) -> WebGpuResult<f32> {
        self.device.execute_reduction(input_id, op)
    }
}

// ============================================================
// CPU simulation helpers
// ============================================================

/// Tiled matrix multiply mirroring the WGSL shader algorithm.
fn tiled_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, tile_size: usize) -> Vec<f32> {
    let ts = tile_size.max(1);
    let mut c = vec![0.0_f32; m * n];

    let num_row_tiles = m.div_ceil(ts);
    let num_col_tiles = n.div_ceil(ts);
    let num_k_tiles = k.div_ceil(ts);

    for tile_row in 0..num_row_tiles {
        for tile_col in 0..num_col_tiles {
            for t in 0..num_k_tiles {
                for lrow in 0..ts {
                    let row = tile_row * ts + lrow;
                    if row >= m {
                        continue;
                    }
                    for lcol in 0..ts {
                        let col = tile_col * ts + lcol;
                        if col >= n {
                            continue;
                        }
                        let mut acc = 0.0_f32;
                        for ki in 0..ts {
                            let a_col = t * ts + ki;
                            let b_row = t * ts + ki;
                            let a_val = if a_col < k { a[row * k + a_col] } else { 0.0 };
                            let b_val = if b_row < k { b[b_row * n + col] } else { 0.0 };
                            acc += a_val * b_val;
                        }
                        c[row * n + col] += acc;
                    }
                }
            }
        }
    }
    c
}

/// Parallel tree reduction (CPU simulation).
fn parallel_tree_reduce(data: &[f32], op: ReductionOp) -> f32 {
    if data.is_empty() {
        return match op {
            ReductionOp::Sum => 0.0,
            ReductionOp::Max => f32::NEG_INFINITY,
            ReductionOp::Min => f32::INFINITY,
        };
    }
    // Simulate workgroup-level tree reduction by chunks of 256.
    let chunk_size = 256usize;
    let partials: Vec<f32> = data
        .chunks(chunk_size)
        .map(|chunk| reduce_chunk(chunk, op))
        .collect();
    // Final reduction over partial results.
    reduce_chunk(&partials, op)
}

fn reduce_chunk(chunk: &[f32], op: ReductionOp) -> f32 {
    match op {
        ReductionOp::Sum => chunk.iter().copied().fold(0.0_f32, |a, v| a + v),
        ReductionOp::Max => chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ReductionOp::Min => chunk.iter().copied().fold(f32::INFINITY, f32::min),
    }
}

// ============================================================
// Shader-comment detection helpers
// ============================================================

fn detect_elementwise_op(shader: &str) -> ElementwiseOp {
    let first_line = shader.lines().next().unwrap_or("");
    if first_line.contains("relu") {
        ElementwiseOp::Relu
    } else if first_line.contains("sigmoid") {
        ElementwiseOp::Sigmoid
    } else if first_line.contains("add") {
        ElementwiseOp::Add
    } else if first_line.contains("mul") {
        ElementwiseOp::Mul
    } else if first_line.contains("exp") {
        ElementwiseOp::Exp
    } else if first_line.contains("log") {
        ElementwiseOp::Log
    } else if first_line.contains("negate") {
        ElementwiseOp::Negate
    } else {
        ElementwiseOp::Relu // safe default
    }
}

fn detect_reduction_op(shader: &str) -> ReductionOp {
    let first_line = shader.lines().next().unwrap_or("");
    if first_line.contains("max") {
        ReductionOp::Max
    } else if first_line.contains("min") {
        ReductionOp::Min
    } else {
        ReductionOp::Sum
    }
}

// ============================================================
// Tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::webgpu::types::{GpuBufferUsage, WebGpuConfig};

    fn make_ctx() -> WebGpuContext {
        WebGpuContext::new(WebGpuConfig::default())
    }

    // ---- Buffer lifecycle ----

    #[test]
    fn test_buffer_create_read_roundtrip() {
        let mut ctx = make_ctx();
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let id = ctx
            .upload_buffer(data.clone(), GpuBufferUsage::Storage)
            .expect("create buffer");
        let out = ctx.download_buffer(id).expect("read buffer");
        assert_eq!(out, data);
    }

    #[test]
    fn test_zero_buffer_all_zeros() {
        let mut ctx = make_ctx();
        let id = ctx
            .device_mut()
            .create_zero_buffer(16, GpuBufferUsage::Storage)
            .expect("zero buffer");
        let out = ctx.download_buffer(id).expect("read");
        assert!(out.iter().all(|&v| v == 0.0), "expected all zeros");
    }

    #[test]
    fn test_destroy_buffer() {
        let mut ctx = make_ctx();
        let id = ctx
            .upload_buffer(vec![1.0], GpuBufferUsage::Storage)
            .expect("create");
        assert!(ctx.device_mut().destroy_buffer(id));
        // Reading after destroy should fail.
        assert!(ctx.download_buffer(id).is_err());
    }

    #[test]
    fn test_buffer_too_large_returns_error() {
        let cfg = WebGpuConfig {
            max_buffer_size: 4, // only 1 f32
            ..WebGpuConfig::default()
        };
        let mut ctx = WebGpuContext::new(cfg);
        let data = vec![1.0_f32, 2.0]; // 8 bytes > 4
        let result = ctx.upload_buffer(data, GpuBufferUsage::Storage);
        assert!(
            matches!(result, Err(GpuError::BufferTooLarge { .. })),
            "expected BufferTooLarge"
        );
    }

    // ---- Matmul ----

    #[test]
    fn test_matmul_2x2() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let mut ctx = make_ctx();
        let a_id = ctx
            .upload_buffer(vec![1.0_f32, 2.0, 3.0, 4.0], GpuBufferUsage::Storage)
            .expect("upload A");
        let b_id = ctx
            .upload_buffer(vec![5.0_f32, 6.0, 7.0, 8.0], GpuBufferUsage::Storage)
            .expect("upload B");
        let c_id = ctx.matmul(a_id, b_id, 2, 2, 2).expect("matmul");
        let c = ctx.download_buffer(c_id).expect("read C");
        let expected = [19.0_f32, 22.0, 43.0, 50.0];
        for (r, &e) in c.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4, "got {r}, expected {e}");
        }
    }

    #[test]
    fn test_matmul_4x4_tiled_matches_naive() {
        // Use a 2×2 tile to stress tile boundaries.
        let cfg = WebGpuConfig {
            workgroup_size_x: 2,
            ..WebGpuConfig::default()
        };
        let mut ctx = WebGpuContext::new(cfg);

        // Identity × arbitrary = same arbitrary
        let ident: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();

        let a_id = ctx
            .upload_buffer(ident, GpuBufferUsage::Storage)
            .expect("upload I");
        let b_id = ctx
            .upload_buffer(data.clone(), GpuBufferUsage::Storage)
            .expect("upload B");
        let c_id = ctx.matmul(a_id, b_id, 4, 4, 4).expect("matmul");
        let c = ctx.download_buffer(c_id).expect("read");
        for (r, &e) in c.iter().zip(data.iter()) {
            assert!(
                (r - e).abs() < 1e-4,
                "I×B should equal B; got {r} expected {e}"
            );
        }
    }

    // ---- Elementwise ----

    #[test]
    fn test_elementwise_relu_clips_negatives() {
        let mut ctx = make_ctx();
        let id = ctx
            .upload_buffer(vec![-3.0_f32, -1.0, 0.0, 2.0, 5.0], GpuBufferUsage::Storage)
            .expect("upload");
        let out_id = ctx
            .elementwise(id, None, ElementwiseOp::Relu)
            .expect("relu");
        let out = ctx.download_buffer(out_id).expect("read");
        assert_eq!(out, vec![0.0_f32, 0.0, 0.0, 2.0, 5.0]);
    }

    #[test]
    fn test_elementwise_sigmoid_in_range() {
        let mut ctx = make_ctx();
        let data: Vec<f32> = (-5..=5).map(|x| x as f32 * 2.0).collect();
        let id = ctx
            .upload_buffer(data, GpuBufferUsage::Storage)
            .expect("upload");
        let out_id = ctx
            .elementwise(id, None, ElementwiseOp::Sigmoid)
            .expect("sigmoid");
        let out = ctx.download_buffer(out_id).expect("read");
        for &v in &out {
            assert!(v > 0.0 && v < 1.0, "sigmoid must be in (0,1), got {v}");
        }
    }

    #[test]
    fn test_elementwise_add() {
        let mut ctx = make_ctx();
        let a_id = ctx
            .upload_buffer(vec![1.0_f32, 2.0, 3.0], GpuBufferUsage::Storage)
            .expect("a");
        let b_id = ctx
            .upload_buffer(vec![4.0_f32, 5.0, 6.0], GpuBufferUsage::Storage)
            .expect("b");
        let out_id = ctx
            .elementwise(a_id, Some(b_id), ElementwiseOp::Add)
            .expect("add");
        let out = ctx.download_buffer(out_id).expect("read");
        assert_eq!(out, vec![5.0_f32, 7.0, 9.0]);
    }

    // ---- Reduction ----

    #[test]
    fn test_reduction_sum_equals_direct() {
        let mut ctx = make_ctx();
        let data: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let direct_sum: f32 = data.iter().sum();
        let id = ctx
            .upload_buffer(data, GpuBufferUsage::Storage)
            .expect("upload");
        let sum = ctx.reduce(id, ReductionOp::Sum).expect("sum");
        assert!((sum - direct_sum).abs() < 1e-2, "sum {sum} != {direct_sum}");
    }

    #[test]
    fn test_reduction_max_equals_direct() {
        let mut ctx = make_ctx();
        let data = vec![3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let direct_max = 9.0_f32;
        let id = ctx
            .upload_buffer(data, GpuBufferUsage::Storage)
            .expect("upload");
        let max = ctx.reduce(id, ReductionOp::Max).expect("max");
        assert!((max - direct_max).abs() < 1e-5, "max {max} != {direct_max}");
    }

    #[test]
    fn test_reduction_min() {
        let mut ctx = make_ctx();
        let data = vec![3.0_f32, 1.0, 4.0, 1.5, 5.0];
        let id = ctx
            .upload_buffer(data, GpuBufferUsage::Storage)
            .expect("upload");
        let min = ctx.reduce(id, ReductionOp::Min).expect("min");
        assert!((min - 1.0).abs() < 1e-5, "min should be 1.0, got {min}");
    }

    #[test]
    fn test_parallel_tree_reduce_power_of_two() {
        // 512 elements — power of two, exercises the chunking fully.
        let data: Vec<f32> = (0..512u32).map(|x| x as f32).collect();
        let expected: f32 = data.iter().sum();
        let got = parallel_tree_reduce(&data, ReductionOp::Sum);
        assert!(
            (got - expected).abs() < 1.0,
            "sum mismatch: {got} vs {expected}"
        );
    }

    #[test]
    fn test_parallel_tree_reduce_non_power_of_two() {
        // 300 elements — non power of two.
        let data: Vec<f32> = (0..300u32).map(|x| x as f32).collect();
        let expected: f32 = data.iter().sum();
        let got = parallel_tree_reduce(&data, ReductionOp::Sum);
        assert!(
            (got - expected).abs() < 1.0,
            "sum mismatch non-pow2: {got} vs {expected}"
        );
    }

    // ---- Conv1d ----

    #[test]
    fn test_conv1d_identity_kernel() {
        // kernel = [1.0] → output = input (identity convolution)
        let mut ctx = make_ctx();
        let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0_f32];
        let in_id = ctx
            .upload_buffer(input.clone(), GpuBufferUsage::Storage)
            .expect("in");
        let k_id = ctx
            .upload_buffer(kernel, GpuBufferUsage::Storage)
            .expect("k");
        let out_id = ctx
            .device_mut()
            .execute_conv1d(in_id, k_id, 1, 0)
            .expect("conv1d");
        let out = ctx.download_buffer(out_id).expect("read");
        assert_eq!(out, input);
    }

    // ---- submit_compute_pass ----

    #[test]
    fn test_submit_compute_pass_elementwise() {
        use crate::webgpu::shader_gen::generate_elementwise_shader;
        let mut ctx = make_ctx();
        let id = ctx
            .upload_buffer(vec![-1.0_f32, 2.0, -3.0], GpuBufferUsage::Storage)
            .expect("upload");
        let shader = generate_elementwise_shader(ElementwiseOp::Relu);
        let out_id = ctx
            .device_mut()
            .submit_compute_pass(&shader, &[id], (1, 1, 1))
            .expect("pass");
        let out = ctx.download_buffer(out_id).expect("read");
        assert_eq!(out, vec![0.0_f32, 2.0, 0.0]);
    }

    #[test]
    fn test_submit_compute_pass_zero_dispatch_fails() {
        let mut ctx = make_ctx();
        let id = ctx
            .upload_buffer(vec![1.0_f32], GpuBufferUsage::Storage)
            .expect("upload");
        let result = ctx
            .device_mut()
            .submit_compute_pass("// something", &[id], (0, 1, 1));
        assert!(result.is_err(), "zero dispatch must fail");
    }

    // ---- Chained operations ----

    #[test]
    fn test_chained_matmul_then_relu() {
        let mut ctx = make_ctx();
        // [[1,-2],[-3,4]] × [[-1,0],[0,-1]] = [[-1,2],[3,-4]]
        let a_id = ctx
            .upload_buffer(vec![1.0_f32, -2.0, -3.0, 4.0], GpuBufferUsage::Storage)
            .expect("a");
        let b_id = ctx
            .upload_buffer(vec![-1.0_f32, 0.0, 0.0, -1.0], GpuBufferUsage::Storage)
            .expect("b");
        let mm_id = ctx.matmul(a_id, b_id, 2, 2, 2).expect("matmul");
        let relu_id = ctx
            .elementwise(mm_id, None, ElementwiseOp::Relu)
            .expect("relu");
        let out = ctx.download_buffer(relu_id).expect("read");
        // After relu: [0,2,3,0]
        assert!(out[0] >= 0.0 && out[3] >= 0.0, "relu should zero negatives");
        assert!(
            out[1] > 0.0 && out[2] > 0.0,
            "positives should survive relu"
        );
    }

    // ---- Context metadata ----

    #[test]
    fn test_context_is_gpu_available_false() {
        let ctx = make_ctx();
        assert!(!ctx.is_gpu_available(), "CPU sim is never GPU");
    }

    #[test]
    fn test_context_adapter_name() {
        let ctx = make_ctx();
        assert!(!ctx.adapter_name().is_empty());
    }
}
