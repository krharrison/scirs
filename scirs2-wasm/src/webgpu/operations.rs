//! Element-wise and reduction WebGPU operations.
//!
//! Each operation provides:
//! * A CPU implementation for use in non-browser contexts and testing.
//! * A `generate_*_shader()` helper that returns WGSL source for GPU execution.

use crate::webgpu::types::WebGpuConfig;

/// Collection of element-wise and reduction operations backed by the WebGPU
/// backend (or a CPU fallback with matching behaviour).
pub struct WebGpuOps {
    config: WebGpuConfig,
}

impl WebGpuOps {
    /// Construct `WebGpuOps` from the given configuration.
    pub fn new(config: WebGpuConfig) -> Self {
        Self { config }
    }

    /// Return the active configuration.
    pub fn config(&self) -> &WebGpuConfig {
        &self.config
    }

    // ------------------------------------------------------------------
    // Element-wise operations (CPU)
    // ------------------------------------------------------------------

    /// Element-wise addition: `c[i] = a[i] + b[i]`.
    ///
    /// # Panics (debug only)
    /// If `a.len() != b.len()`.
    pub fn add_f32(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        debug_assert_eq!(a.len(), b.len(), "add_f32: length mismatch");
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    /// Element-wise multiplication: `c[i] = a[i] * b[i]`.
    ///
    /// # Panics (debug only)
    /// If `a.len() != b.len()`.
    pub fn mul_f32(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        debug_assert_eq!(a.len(), b.len(), "mul_f32: length mismatch");
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    /// Element-wise ReLU: `c[i] = max(0, a[i])`.
    pub fn relu_f32(&self, a: &[f32]) -> Vec<f32> {
        a.iter().map(|&x| x.max(0.0)).collect()
    }

    /// Element-wise sigmoid: `c[i] = 1 / (1 + exp(-a[i]))`.
    pub fn sigmoid_f32(&self, a: &[f32]) -> Vec<f32> {
        a.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    /// Numerically stable softmax over the entire slice.
    ///
    /// Subtracts `max(a)` before exponentiation to avoid overflow, then divides
    /// each element by the sum.
    pub fn softmax_f32(&self, a: &[f32]) -> Vec<f32> {
        if a.is_empty() {
            return Vec::new();
        }
        // Find max for numerical stability.
        let max_val = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = a.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 {
            // Degenerate case: all elements are -∞ → uniform distribution.
            let n = a.len() as f32;
            return vec![1.0 / n; a.len()];
        }
        exps.iter().map(|&e| e / sum).collect()
    }

    // ------------------------------------------------------------------
    // Reduction operations (CPU)
    // ------------------------------------------------------------------

    /// Sum all elements of `a`.
    pub fn reduce_sum_f32(&self, a: &[f32]) -> f32 {
        a.iter().copied().fold(0.0_f32, |acc, x| acc + x)
    }

    /// Return the maximum element of `a`, or `f32::NEG_INFINITY` for empty slices.
    pub fn reduce_max_f32(&self, a: &[f32]) -> f32 {
        a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    // ------------------------------------------------------------------
    // WGSL shader generation
    // ------------------------------------------------------------------

    /// Generate a WGSL compute shader for a named element-wise operation.
    ///
    /// Supported `op` values: `"add"`, `"mul"`, `"relu"`, `"sigmoid"`, `"negate"`.
    /// Any other value produces a passthrough (identity) shader.
    ///
    /// The returned shader expects:
    /// * Binding 0 — input buffer A (`array<f32>`)
    /// * Binding 1 — input buffer B (`array<f32>`) — ignored for unary ops
    /// * Binding 2 — output buffer (`array<f32>`)
    /// * Binding 3 — uniform containing the element count as `u32`
    pub fn generate_elementwise_shader(&self, op: &str) -> String {
        let body = match op {
            "add" => "output[i] = a[i] + b[i];".to_owned(),
            "mul" => "output[i] = a[i] * b[i];".to_owned(),
            "relu" => "output[i] = max(a[i], 0.0);".to_owned(),
            "sigmoid" => "output[i] = 1.0 / (1.0 + exp(-a[i]));".to_owned(),
            "negate" => "output[i] = -a[i];".to_owned(),
            _ => "output[i] = a[i];".to_owned(), // identity / passthrough
        };

        format!(
            r#"// Element-wise operation: {op}

@group(0) @binding(0) var<storage, read>       a      : array<f32>;
@group(0) @binding(1) var<storage, read>       b      : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform>             n      : u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= n) {{ return; }}
    {body}
}}
"#,
            op = op,
            body = body,
        )
    }

    /// Generate a WGSL compute shader for a parallel reduction (sum or max).
    ///
    /// Uses a two-pass workgroup-local tree reduction followed by an atomic
    /// write to the output.
    ///
    /// Supported `op` values: `"sum"`, `"max"`.
    pub fn generate_reduction_shader(&self, op: &str) -> String {
        let (init, combine, atomic_fn) = match op {
            "max" => ("f32(-3.40282347e+38)", "max(acc, val)", "atomicMax"),
            _ /* "sum" */ => ("0.0", "acc + val", "atomicAdd"),
        };

        format!(
            r#"// Parallel reduction: {op}
// NOTE: WebGPU does not have atomicAdd/Max on f32 natively;
//       this shader uses a two-step approach via bitcast and
//       is provided as a reference template only.

@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<atomic<i32>>;
@group(0) @binding(2) var<uniform>             n      : u32;

var<workgroup> local_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {{
    let i = gid.x;
    var acc: f32 = {init};
    if (i < n) {{
        acc = input[i];
    }}
    local_data[lid.x] = acc;
    workgroupBarrier();

    // Tree reduction within workgroup
    var stride: u32 = 128u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            let val = local_data[lid.x + stride];
            local_data[lid.x] = {combine};
        }}
        workgroupBarrier();
        stride = stride >> 1u;
    }}

    if (lid.x == 0u) {{
        // Write partial result (bitcast float to int for atomic store)
        {atomic_fn}(&output[0], bitcast<i32>(local_data[0]));
    }}
}}
"#,
            op = op,
            init = init,
            combine = combine,
            atomic_fn = atomic_fn,
        )
    }
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn ops() -> WebGpuOps {
        WebGpuOps::new(WebGpuConfig::default())
    }

    #[test]
    fn test_sigmoid_output_in_range() {
        let o = ops();
        let input: Vec<f32> = (-10..=10).map(|x| x as f32).collect();
        let out = o.sigmoid_f32(&input);
        for &v in &out {
            assert!(v > 0.0 && v < 1.0, "sigmoid out of (0,1): {v}");
        }
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let o = ops();
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = o.softmax_f32(&input);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
    }

    #[test]
    fn test_relu_zeroes_negatives() {
        let o = ops();
        let input = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let out = o.relu_f32(&input);
        let expected = [0.0_f32, 0.0, 0.0, 1.0, 2.0];
        for (r, e) in out.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reduce_sum() {
        let o = ops();
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert!((o.reduce_sum_f32(&data) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_max() {
        let o = ops();
        let data = vec![3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0];
        assert!((o.reduce_max_f32(&data) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_elementwise_shader_add() {
        let o = ops();
        let shader = o.generate_elementwise_shader("add");
        assert!(
            shader.contains("workgroup_size"),
            "shader must have workgroup_size"
        );
        assert!(shader.contains("add"), "shader must mention the op name");
    }

    #[test]
    fn test_add_and_mul_f32() {
        let o = ops();
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        let add = o.add_f32(&a, &b);
        let mul = o.mul_f32(&a, &b);
        assert_eq!(add, vec![5.0, 7.0, 9.0]);
        assert_eq!(mul, vec![4.0, 10.0, 18.0]);
    }
}
