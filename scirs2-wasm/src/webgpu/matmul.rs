//! GPU-accelerated matrix multiply for the WebGPU backend.
//!
//! Two paths are provided:
//!
//! 1. **WGSL shader generation** — `generate_wgsl_shader()` returns a complete,
//!    valid WGSL compute shader string that can be compiled by a `GPUDevice` in the
//!    browser.  The shader uses a tiled algorithm where each workgroup loads a
//!    `(tile_size × tile_size)` sub-matrix of A and B into workgroup-shared memory
//!    before computing partial dot products.
//!
//! 2. **CPU simulation** — `matmul_tiled()` is a pure-Rust implementation that
//!    mirrors the same tiling strategy, making it straightforward to compare Rust
//!    and shader outputs during development and testing.

use crate::webgpu::types::WebGpuConfig;

/// Performs tiled matrix multiplication, either via WGSL (browser) or CPU simulation.
pub struct WebGpuMatmul {
    config: WebGpuConfig,
}

impl WebGpuMatmul {
    /// Construct a new `WebGpuMatmul` from `config`.
    pub fn new(config: WebGpuConfig) -> Self {
        Self { config }
    }

    /// Return the active [`WebGpuConfig`].
    pub fn config(&self) -> &WebGpuConfig {
        &self.config
    }

    // ------------------------------------------------------------------
    // WGSL shader generation
    // ------------------------------------------------------------------

    /// Generate a complete WGSL compute shader for tiled matrix multiplication.
    ///
    /// The returned string can be passed to `GPUDevice.createShaderModule()`.
    ///
    /// # Shader layout
    /// * Binding 0 — matrix A  (`array<f32>`, row-major, shape M×K)
    /// * Binding 1 — matrix B  (`array<f32>`, row-major, shape K×N)
    /// * Binding 2 — output C  (`array<f32>`, row-major, shape M×N)
    /// * Binding 3 — uniforms  (`vec3<u32>` containing M, K, N)
    ///
    /// Each workgroup covers a `(tile_size × tile_size)` output tile.
    pub fn generate_wgsl_shader(&self, tile_size: usize) -> String {
        let ts = tile_size;
        format!(
            r#"// Tiled matrix multiplication — tile size {ts}x{ts}
// Binding layout (group 0):
//   0 : read-only storage  -> matrix A  [M*K f32, row-major]
//   1 : read-only storage  -> matrix B  [K*N f32, row-major]
//   2 : read-write storage -> output C  [M*N f32, row-major]
//   3 : uniform            -> dims      [M: u32, K: u32, N: u32]

struct Dims {{
    M: u32,
    K: u32,
    N: u32,
}};

@group(0) @binding(0) var<storage, read>       matA  : array<f32>;
@group(0) @binding(1) var<storage, read>       matB  : array<f32>;
@group(0) @binding(2) var<storage, read_write> matC  : array<f32>;
@group(0) @binding(3) var<uniform>             dims  : Dims;

const TILE_SIZE: u32 = {ts}u;

var<workgroup> tileA : array<f32, {tile_sq}>;
var<workgroup> tileB : array<f32, {tile_sq}>;

@compute @workgroup_size({ts}, {ts})
fn main(
    @builtin(global_invocation_id)   global_id   : vec3<u32>,
    @builtin(local_invocation_id)    local_id    : vec3<u32>,
    @builtin(workgroup_id)           workgroup_id: vec3<u32>,
) {{
    let row = global_id.y;
    let col = global_id.x;
    let lrow = local_id.y;
    let lcol = local_id.x;

    var acc: f32 = 0.0;

    let num_tiles: u32 = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t++) {{
        // Load tile of A into shared memory
        let a_col = t * TILE_SIZE + lcol;
        if (row < dims.M && a_col < dims.K) {{
            tileA[lrow * TILE_SIZE + lcol] = matA[row * dims.K + a_col];
        }} else {{
            tileA[lrow * TILE_SIZE + lcol] = 0.0;
        }}

        // Load tile of B into shared memory
        let b_row = t * TILE_SIZE + lrow;
        if (b_row < dims.K && col < dims.N) {{
            tileB[lrow * TILE_SIZE + lcol] = matB[b_row * dims.N + col];
        }} else {{
            tileB[lrow * TILE_SIZE + lcol] = 0.0;
        }}

        workgroupBarrier();

        // Compute partial dot product for this tile
        for (var k: u32 = 0u; k < TILE_SIZE; k++) {{
            acc += tileA[lrow * TILE_SIZE + k] * tileB[k * TILE_SIZE + lcol];
        }}

        workgroupBarrier();
    }}

    if (row < dims.M && col < dims.N) {{
        matC[row * dims.N + col] = acc;
    }}
}}
"#,
            ts = ts,
            tile_sq = ts * ts,
        )
    }

    // ------------------------------------------------------------------
    // CPU simulation (tiled, mirrors WGSL algorithm)
    // ------------------------------------------------------------------

    /// CPU-side tiled matrix multiplication.
    ///
    /// Computes `C = A × B` where:
    /// * `a` is an `m × k` matrix stored row-major.
    /// * `b` is a `k × n` matrix stored row-major.
    /// * Returns an `m × n` matrix stored row-major.
    ///
    /// The implementation processes `(tile_size × tile_size)` tiles in the same
    /// order as the WGSL shader, making numerical comparison straightforward.
    pub fn matmul_tiled(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        debug_assert_eq!(a.len(), m * k, "matrix A must have exactly m*k elements");
        debug_assert_eq!(b.len(), k * n, "matrix B must have exactly k*n elements");

        let ts = self.config.tile_size;
        let mut c = vec![0.0_f32; m * n];

        let num_tiles_k = k.div_ceil(ts);

        // Iterate over output tiles (row-tile, col-tile)
        let num_row_tiles = m.div_ceil(ts);
        let num_col_tiles = n.div_ceil(ts);

        for tile_row in 0..num_row_tiles {
            for tile_col in 0..num_col_tiles {
                // For each k-tile
                for t in 0..num_tiles_k {
                    // Simulate workgroup shared-memory load + compute
                    // for every thread (lrow, lcol) in the workgroup.
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
                            // Load tileA[lrow, k_idx] and tileB[k_idx, lcol]
                            // and accumulate.
                            let mut acc = 0.0_f32;
                            for k_idx in 0..ts {
                                let a_col = t * ts + k_idx;
                                let b_row = t * ts + k_idx;

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

    /// Batched matrix multiply: computes `C[i] = A[i] × B[i]` for
    /// `i in 0..batch`.
    ///
    /// * `a` — flat buffer, shape `(batch, m, k)` in row-major order.
    /// * `b` — flat buffer, shape `(batch, k, n)` in row-major order.
    /// * Returns a flat buffer of shape `(batch, m, n)` in row-major order.
    pub fn matmul_batch(
        &self,
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        debug_assert_eq!(a.len(), batch * m * k);
        debug_assert_eq!(b.len(), batch * k * n);

        let stride_a = m * k;
        let stride_b = k * n;
        let stride_c = m * n;
        let mut out = Vec::with_capacity(batch * stride_c);

        for i in 0..batch {
            let a_slice = &a[i * stride_a..(i + 1) * stride_a];
            let b_slice = &b[i * stride_b..(i + 1) * stride_b];
            let c_slice = self.matmul_tiled(a_slice, b_slice, m, k, n);
            out.extend_from_slice(&c_slice);
        }

        out
    }
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn default_matmul() -> WebGpuMatmul {
        WebGpuMatmul::new(WebGpuConfig::default())
    }

    #[test]
    fn test_matmul_identity() {
        // I × I = I  (3×3 identity)
        let mm = default_matmul();
        let identity: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = mm.matmul_tiled(&identity, &identity, 3, 3, 3);
        for (r, e) in result.iter().zip(identity.iter()) {
            assert!((r - e).abs() < 1e-6, "identity matmul failed: {r} != {e}");
        }
    }

    #[test]
    fn test_matmul_2x2() {
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let mm = default_matmul();
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let c = mm.matmul_tiled(&a, &b, 2, 2, 2);
        let expected = [19.0_f32, 22.0, 43.0, 50.0];
        for (r, e) in c.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "2×2 matmul: got {r}, expected {e}");
        }
    }

    #[test]
    fn test_generate_wgsl_shader_non_empty_contains_workgroup() {
        let mm = default_matmul();
        let shader = mm.generate_wgsl_shader(16);
        assert!(!shader.is_empty(), "shader must be non-empty");
        assert!(
            shader.contains("workgroup"),
            "shader must reference 'workgroup'"
        );
    }

    #[test]
    fn test_batched_matmul_output_shape() {
        let mm = default_matmul();
        let batch = 3;
        let (m, k, n) = (2, 3, 4);
        let a = vec![1.0_f32; batch * m * k];
        let b = vec![1.0_f32; batch * k * n];
        let c = mm.matmul_batch(&a, &b, batch, m, k, n);
        assert_eq!(c.len(), batch * m * n);
    }
}
