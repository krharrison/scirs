//! WGSL shader code generation for common GPU operations.
//!
//! Provides a `WgslGenerator` struct with method chaining and standalone
//! `generate_*` free functions that emit complete, valid WGSL compute shaders
//! for use in browser environments via the WebGPU API.

// ============================================================
// Enumerations
// ============================================================

/// Describes an element-wise operation for shader generation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseOp {
    /// `c[i] = a[i] + b[i]`
    Add,
    /// `c[i] = a[i] * b[i]`
    Mul,
    /// `c[i] = max(0.0, a[i])`
    Relu,
    /// `c[i] = 1 / (1 + exp(-a[i]))`
    Sigmoid,
    /// `c[i] = exp(a[i])`
    Exp,
    /// `c[i] = log(a[i])`
    Log,
    /// `c[i] = -a[i]`
    Negate,
}

impl ElementwiseOp {
    /// Return the WGSL expression body for the operation.
    fn wgsl_body(self) -> &'static str {
        match self {
            Self::Add => "output[i] = a[i] + b[i];",
            Self::Mul => "output[i] = a[i] * b[i];",
            Self::Relu => "output[i] = max(a[i], 0.0);",
            Self::Sigmoid => "output[i] = 1.0 / (1.0 + exp(-a[i]));",
            Self::Exp => "output[i] = exp(a[i]);",
            Self::Log => "output[i] = log(a[i]);",
            Self::Negate => "output[i] = -a[i];",
        }
    }

    /// Human-readable label used in shader comments.
    fn label(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Negate => "negate",
        }
    }
}

/// Describes a reduction operation for shader generation.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Compute the sum of all elements.
    Sum,
    /// Compute the maximum element.
    Max,
    /// Compute the minimum element.
    Min,
}

impl ReductionOp {
    fn label(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Max => "max",
            Self::Min => "min",
        }
    }

    fn init_value(self) -> &'static str {
        match self {
            Self::Sum => "0.0",
            Self::Max => "f32(-3.40282347e+38)",
            Self::Min => "f32(3.40282347e+38)",
        }
    }

    fn combine_expr(self) -> &'static str {
        match self {
            Self::Sum => "local_data[lid.x] = local_data[lid.x] + val;",
            Self::Max => "local_data[lid.x] = max(local_data[lid.x], val);",
            Self::Min => "local_data[lid.x] = min(local_data[lid.x], val);",
        }
    }
}

// ============================================================
// Free-function generators (mirrors WgslGenerator methods)
// ============================================================

/// Generate a tiled GEMM compute shader.
///
/// # Arguments
/// * `m`, `n`, `k` – logical matrix dimensions (used in comments only; the
///   shader accepts them as runtime uniforms).
/// * `tile_size` – workgroup tile side length (typically 8 or 16).
///
/// # Binding layout (group 0)
/// | Binding | Role |
/// |---------|------|
/// | 0 | A  (storage, read) — `array<f32>`, shape M×K row-major |
/// | 1 | B  (storage, read) — `array<f32>`, shape K×N row-major |
/// | 2 | C  (storage, read\_write) — `array<f32>`, shape M×N row-major |
/// | 3 | dims (uniform) — `struct { M: u32, K: u32, N: u32 }` |
pub fn generate_matmul_shader(m: usize, n: usize, k: usize, tile_size: usize) -> String {
    let ts = tile_size;
    let tile_sq = ts * ts;
    format!(
        r#"// Tiled GEMM — logical shape ({m}×{k}) × ({k}×{n}), tile {ts}×{ts}
// group 0:
//   binding 0 — matA [M*K f32, row-major]
//   binding 1 — matB [K*N f32, row-major]
//   binding 2 — matC [M*N f32, row-major]  (output)
//   binding 3 — Dims uniform {{ M: u32, K: u32, N: u32 }}

struct Dims {{
    M: u32,
    K: u32,
    N: u32,
}};

@group(0) @binding(0) var<storage, read>       matA : array<f32>;
@group(0) @binding(1) var<storage, read>       matB : array<f32>;
@group(0) @binding(2) var<storage, read_write> matC : array<f32>;
@group(0) @binding(3) var<uniform>             dims : Dims;

const TILE_SIZE: u32 = {ts}u;

var<workgroup> tileA: array<f32, {tile_sq}>;
var<workgroup> tileB: array<f32, {tile_sq}>;

@compute @workgroup_size({ts}, {ts}, 1)
fn main(
    @builtin(global_invocation_id)  global_id   : vec3<u32>,
    @builtin(local_invocation_id)   local_id    : vec3<u32>,
) {{
    let row  = global_id.y;
    let col  = global_id.x;
    let lrow = local_id.y;
    let lcol = local_id.x;

    var acc: f32 = 0.0;

    let num_tiles: u32 = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t++) {{
        let a_col = t * TILE_SIZE + lcol;
        if (row < dims.M && a_col < dims.K) {{
            tileA[lrow * TILE_SIZE + lcol] = matA[row * dims.K + a_col];
        }} else {{
            tileA[lrow * TILE_SIZE + lcol] = 0.0;
        }}

        let b_row = t * TILE_SIZE + lrow;
        if (b_row < dims.K && col < dims.N) {{
            tileB[lrow * TILE_SIZE + lcol] = matB[b_row * dims.N + col];
        }} else {{
            tileB[lrow * TILE_SIZE + lcol] = 0.0;
        }}

        workgroupBarrier();

        for (var ki: u32 = 0u; ki < TILE_SIZE; ki++) {{
            acc += tileA[lrow * TILE_SIZE + ki] * tileB[ki * TILE_SIZE + lcol];
        }}

        workgroupBarrier();
    }}

    if (row < dims.M && col < dims.N) {{
        matC[row * dims.N + col] = acc;
    }}
}}
"#,
        m = m,
        n = n,
        k = k,
        ts = ts,
        tile_sq = tile_sq,
    )
}

/// Generate an element-wise compute shader.
///
/// # Binding layout (group 0)
/// | Binding | Role |
/// |---------|------|
/// | 0 | a (storage, read) |
/// | 1 | b (storage, read) — ignored for unary ops |
/// | 2 | output (storage, read\_write) |
/// | 3 | n (uniform, u32) — element count |
pub fn generate_elementwise_shader(op: ElementwiseOp) -> String {
    format!(
        r#"// Element-wise operation: {label}

@group(0) @binding(0) var<storage, read>       a      : array<f32>;
@group(0) @binding(1) var<storage, read>       b      : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform>             n      : u32;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= n) {{ return; }}
    {body}
}}
"#,
        label = op.label(),
        body = op.wgsl_body(),
    )
}

/// Generate a parallel tree-reduction compute shader.
///
/// Uses a workgroup-local scratchpad array of size 256 (matching
/// `@workgroup_size(256)`).  The partial result per workgroup is written to
/// `partial[workgroup_id.x]`; the caller must accumulate these partial results.
///
/// # Binding layout (group 0)
/// | Binding | Role |
/// |---------|------|
/// | 0 | input (storage, read) |
/// | 1 | partial (storage, read\_write) — length = number of dispatched workgroups |
/// | 2 | n (uniform, u32) — element count |
pub fn generate_reduction_shader(op: ReductionOp) -> String {
    format!(
        r#"// Parallel tree reduction: {label}

@group(0) @binding(0) var<storage, read>       input   : array<f32>;
@group(0) @binding(1) var<storage, read_write> partial : array<f32>;
@group(0) @binding(2) var<uniform>             n       : u32;

var<workgroup> local_data: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id)  gid  : vec3<u32>,
    @builtin(local_invocation_id)   lid  : vec3<u32>,
    @builtin(workgroup_id)          wgid : vec3<u32>,
) {{
    let i = gid.x;
    var acc: f32 = {init};
    if (i < n) {{
        acc = input[i];
    }}
    local_data[lid.x] = acc;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid.x < stride) {{
            let val = local_data[lid.x + stride];
            {combine}
        }}
        workgroupBarrier();
        stride = stride >> 1u;
    }}

    if (lid.x == 0u) {{
        partial[wgid.x] = local_data[0];
    }}
}}
"#,
        label = op.label(),
        init = op.init_value(),
        combine = op.combine_expr(),
    )
}

/// Generate a 1-D convolution compute shader.
///
/// # Arguments
/// * `kernel_size` – number of taps (must be ≥ 1).
/// * `stride`      – output stride (≥ 1).
/// * `padding`     – zero-padding applied symmetrically to the input.
///
/// # Binding layout (group 0)
/// | Binding | Role |
/// |---------|------|
/// | 0 | input   (storage, read) — length `input_len` |
/// | 1 | kernel  (storage, read) — length `kernel_size` |
/// | 2 | output  (storage, read\_write) |
/// | 3 | params  (uniform) — `{ input_len: u32, output_len: u32 }` |
pub fn generate_conv1d_shader(kernel_size: usize, stride: usize, padding: usize) -> String {
    format!(
        r#"// 1-D convolution — kernel {ks}, stride {st}, padding {pd}

struct Conv1dParams {{
    input_len  : u32,
    output_len : u32,
}};

@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read>       kernel : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform>             params : Conv1dParams;

const KERNEL_SIZE : u32 = {ks}u;
const STRIDE      : u32 = {st}u;
const PADDING     : u32 = {pd}u;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let out_idx = gid.x;
    if (out_idx >= params.output_len) {{ return; }}

    var acc: f32 = 0.0;
    let in_start: i32 = i32(out_idx * STRIDE) - i32(PADDING);

    for (var ki: u32 = 0u; ki < KERNEL_SIZE; ki++) {{
        let in_pos: i32 = in_start + i32(ki);
        if (in_pos >= 0 && u32(in_pos) < params.input_len) {{
            acc += input[u32(in_pos)] * kernel[ki];
        }}
    }}

    output[out_idx] = acc;
}}
"#,
        ks = kernel_size,
        st = stride,
        pd = padding,
    )
}

// ============================================================
// WgslGenerator — builder/method-chaining approach
// ============================================================

/// A builder that generates WGSL compute shaders via method chaining.
///
/// # Example
/// ```
/// use scirs2_wasm::webgpu::shader_gen::{WgslGenerator, ElementwiseOp, ReductionOp};
///
/// let shader = WgslGenerator::new()
///     .with_tile_size(8)
///     .matmul(64, 64, 64);
///
/// assert!(shader.contains("@compute"));
/// ```
#[derive(Debug, Clone)]
pub struct WgslGenerator {
    tile_size: usize,
}

impl Default for WgslGenerator {
    fn default() -> Self {
        Self { tile_size: 16 }
    }
}

impl WgslGenerator {
    /// Create a new generator with default settings (tile\_size = 16).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the workgroup tile size for matrix operations.
    #[must_use]
    pub fn with_tile_size(mut self, size: usize) -> Self {
        self.tile_size = size.max(1);
        self
    }

    /// Generate a tiled GEMM shader.  See [`generate_matmul_shader`].
    pub fn matmul(&self, m: usize, n: usize, k: usize) -> String {
        generate_matmul_shader(m, n, k, self.tile_size)
    }

    /// Generate an element-wise shader.  See [`generate_elementwise_shader`].
    pub fn elementwise(&self, op: ElementwiseOp) -> String {
        generate_elementwise_shader(op)
    }

    /// Generate a parallel tree-reduction shader.  See [`generate_reduction_shader`].
    pub fn reduction(&self, op: ReductionOp) -> String {
        generate_reduction_shader(op)
    }

    /// Generate a 1-D convolution shader.  See [`generate_conv1d_shader`].
    pub fn conv1d(&self, kernel_size: usize, stride: usize, padding: usize) -> String {
        generate_conv1d_shader(kernel_size, stride, padding)
    }
}

// ============================================================
// Tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ---- matmul shader ----

    #[test]
    fn test_matmul_shader_contains_compute_attribute() {
        let s = generate_matmul_shader(4, 4, 4, 16);
        assert!(s.contains("@compute"), "missing @compute");
    }

    #[test]
    fn test_matmul_shader_contains_fn_main() {
        let s = generate_matmul_shader(8, 8, 8, 8);
        assert!(s.contains("fn main"), "missing fn main");
    }

    #[test]
    fn test_matmul_shader_tile_size_reflected() {
        let s = generate_matmul_shader(2, 2, 2, 4);
        assert!(s.contains("4u"), "tile size 4 must appear as 4u");
    }

    #[test]
    fn test_matmul_shader_has_workgroup_barrier() {
        let s = generate_matmul_shader(2, 2, 2, 8);
        assert!(s.contains("workgroupBarrier"), "missing workgroupBarrier");
    }

    // ---- elementwise shader ----

    #[test]
    fn test_elementwise_relu_shader() {
        let s = generate_elementwise_shader(ElementwiseOp::Relu);
        assert!(s.contains("@compute"));
        assert!(s.contains("relu"));
        assert!(s.contains("max(a[i], 0.0)"));
    }

    #[test]
    fn test_elementwise_sigmoid_shader() {
        let s = generate_elementwise_shader(ElementwiseOp::Sigmoid);
        assert!(s.contains("sigmoid"));
        assert!(s.contains("exp(-a[i])"));
    }

    #[test]
    fn test_elementwise_add_shader_has_b_binding() {
        let s = generate_elementwise_shader(ElementwiseOp::Add);
        assert!(s.contains("@binding(1)"), "add needs b binding");
        assert!(s.contains("a[i] + b[i]"));
    }

    #[test]
    fn test_elementwise_exp_shader() {
        let s = generate_elementwise_shader(ElementwiseOp::Exp);
        assert!(s.contains("exp(a[i])"));
    }

    #[test]
    fn test_elementwise_log_shader() {
        let s = generate_elementwise_shader(ElementwiseOp::Log);
        assert!(s.contains("log(a[i])"));
    }

    // ---- reduction shader ----

    #[test]
    fn test_reduction_sum_shader() {
        let s = generate_reduction_shader(ReductionOp::Sum);
        assert!(s.contains("@compute"));
        assert!(s.contains("sum"));
        assert!(s.contains("workgroupBarrier"));
    }

    #[test]
    fn test_reduction_max_shader() {
        let s = generate_reduction_shader(ReductionOp::Max);
        assert!(s.contains("max"));
    }

    #[test]
    fn test_reduction_min_shader() {
        let s = generate_reduction_shader(ReductionOp::Min);
        assert!(s.contains("min"));
    }

    // ---- conv1d shader ----

    #[test]
    fn test_conv1d_shader_contains_compute() {
        let s = generate_conv1d_shader(3, 1, 1);
        assert!(s.contains("@compute"), "missing @compute");
        assert!(s.contains("fn main"), "missing fn main");
    }

    #[test]
    fn test_conv1d_shader_reflects_params() {
        let s = generate_conv1d_shader(5, 2, 2);
        assert!(s.contains("5u"), "kernel size 5 not found");
        assert!(s.contains("STRIDE"), "stride missing");
        assert!(s.contains("PADDING"), "padding missing");
    }

    // ---- WgslGenerator ----

    #[test]
    fn test_wgsl_generator_default_tile_size() {
        let gen = WgslGenerator::new();
        let s = gen.matmul(4, 4, 4);
        assert!(s.contains("16u"), "default tile size 16 should appear");
    }

    #[test]
    fn test_wgsl_generator_custom_tile_size() {
        let gen = WgslGenerator::new().with_tile_size(8);
        let s = gen.matmul(4, 4, 4);
        assert!(s.contains("8u"), "custom tile size 8 should appear");
    }

    #[test]
    fn test_wgsl_generator_chaining() {
        let gen = WgslGenerator::new().with_tile_size(4);
        let mm = gen.matmul(2, 2, 2);
        let ew = gen.elementwise(ElementwiseOp::Relu);
        let rd = gen.reduction(ReductionOp::Sum);
        let cv = gen.conv1d(3, 1, 0);
        assert!(mm.contains("@compute"));
        assert!(ew.contains("@compute"));
        assert!(rd.contains("@compute"));
        assert!(cv.contains("@compute"));
    }

    #[test]
    fn test_wgsl_generator_zero_tile_clamped_to_one() {
        let gen = WgslGenerator::new().with_tile_size(0);
        // Should not panic; tile_size is clamped to 1.
        let s = gen.matmul(1, 1, 1);
        assert!(s.contains("@compute"));
    }
}
