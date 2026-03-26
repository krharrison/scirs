//! Types for the WebGPU backend.

// ============================================================
// WebGpuBackend
// ============================================================

/// Selects the computation backend.
///
/// `WebGpu` is the actual GPU path (available in browsers supporting WebGPU).
/// `Cpu` is a pure-Rust CPU fallback that mirrors the tile algorithm used in the
/// WGSL shaders, making it easy to validate shader correctness.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WebGpuBackend {
    /// Real WebGPU (requires a browser with WebGPU support).
    WebGpu,
    /// Pure-Rust CPU fallback — always available.
    #[default]
    Cpu,
}

// ============================================================
// WebGpuConfig
// ============================================================

/// Configuration for the WebGPU backend.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct WebGpuConfig {
    /// Prefer GPU execution when available.  Default: `true`.
    pub prefer_gpu: bool,
    /// Fall back to CPU when GPU is unavailable.  Default: `true`.
    pub fallback_to_cpu: bool,
    /// Tile size used in matmul workgroups (X dimension).  Default: `16`.
    pub tile_size: usize,
    /// Maximum buffer size in bytes.  Default: 256 MiB.
    pub max_buffer_size: usize,
    /// Workgroup size along the X axis.  Default: `16`.
    pub workgroup_size_x: u32,
    /// Workgroup size along the Y axis.  Default: `16`.
    pub workgroup_size_y: u32,
    /// Workgroup size along the Z axis.  Default: `1`.
    pub workgroup_size_z: u32,
    /// Enable additional validation of buffer sizes and shader parameters.  Default: `true`.
    pub enable_validation: bool,
}

impl Default for WebGpuConfig {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            fallback_to_cpu: true,
            tile_size: 16,
            max_buffer_size: 256 * 1024 * 1024, // 256 MiB
            workgroup_size_x: 16,
            workgroup_size_y: 16,
            workgroup_size_z: 1,
            enable_validation: true,
        }
    }
}

// ============================================================
// GpuBufferUsage
// ============================================================

/// Describes the intended usage of a `GpuBuffer`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuBufferUsage {
    /// General storage buffer used in compute shaders.
    #[default]
    Storage,
    /// Uniform buffer for small constant data (e.g. matrix dimensions).
    Uniform,
    /// Staging buffer used for CPU↔GPU transfers.
    Staging,
    /// Vertex buffer (position/attribute data).
    Vertex,
}

// ============================================================
// GpuBufferDescriptor
// ============================================================

/// Descriptor used when allocating a new `GpuBuffer`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GpuBufferDescriptor {
    /// Total size in bytes.
    pub size: usize,
    /// Intended usage flags.
    pub usage: GpuBufferUsage,
    /// Optional debug label.
    pub label: Option<String>,
}

impl Default for GpuBufferDescriptor {
    fn default() -> Self {
        Self {
            size: 0,
            usage: GpuBufferUsage::Storage,
            label: None,
        }
    }
}

// ============================================================
// ComputePipelineDescriptor
// ============================================================

/// Descriptor for a compute pipeline.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ComputePipelineDescriptor {
    /// WGSL shader source string.
    pub shader_source: String,
    /// Entry-point function name in the shader.  Default: `"main"`.
    pub entry_point: String,
    /// Number of bind group slots required.  Default: `1`.
    pub bind_group_count: u32,
}

impl Default for ComputePipelineDescriptor {
    fn default() -> Self {
        Self {
            shader_source: String::new(),
            entry_point: "main".to_string(),
            bind_group_count: 1,
        }
    }
}

// ============================================================
// GpuError
// ============================================================

/// Error type returned by WebGPU backend operations.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum GpuError {
    /// No WebGPU adapter / device is available.
    DeviceNotAvailable,
    /// Requested buffer size exceeds the device limit.
    BufferTooLarge {
        /// Requested allocation size in bytes.
        requested: usize,
        /// Maximum allowed size in bytes.
        limit: usize,
    },
    /// Shader module compilation failed.
    ShaderCompile(String),
    /// Execution / dispatch error.
    Execution(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeviceNotAvailable => write!(f, "WebGPU device not available"),
            Self::BufferTooLarge { requested, limit } => write!(
                f,
                "buffer size {requested} bytes exceeds limit {limit} bytes"
            ),
            Self::ShaderCompile(msg) => write!(f, "shader compilation error: {msg}"),
            Self::Execution(msg) => write!(f, "GPU execution error: {msg}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Convenience alias for `Result<T, GpuError>`.
pub type WebGpuResult<T> = Result<T, GpuError>;

// ============================================================
// GpuBuffer
// ============================================================

/// A buffer that holds data either on the (simulated) GPU or on the CPU.
///
/// In the current Rust-only implementation the GPU path is represented as a
/// `Vec<f32>` that would be uploaded to a `GPUBuffer` when running inside a
/// browser with WebGPU support.
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// CPU-side mirror of the buffer contents.
    pub data: Vec<f32>,
    /// Number of `f32` elements.
    pub size: usize,
    /// Which backend currently owns the data.
    pub backend: WebGpuBackend,
    /// Usage flags for this buffer.
    pub usage: GpuBufferUsage,
}

impl GpuBuffer {
    /// Allocate a new buffer, filling it with `data`.
    pub fn new(data: Vec<f32>, backend: WebGpuBackend) -> Self {
        let size = data.len();
        Self {
            data,
            size,
            backend,
            usage: GpuBufferUsage::Storage,
        }
    }

    /// Create a buffer with explicit usage.
    pub fn with_data(data: Vec<f32>, usage: GpuBufferUsage) -> Self {
        let size = data.len();
        Self {
            data,
            size,
            backend: WebGpuBackend::Cpu,
            usage,
        }
    }

    /// Create a zero-filled buffer of `size` elements.
    pub fn zeros(size: usize, usage: GpuBufferUsage) -> Self {
        Self {
            data: vec![0.0_f32; size],
            size,
            backend: WebGpuBackend::Cpu,
            usage,
        }
    }

    /// Return a slice of the underlying data.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Return a mutable slice of the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

// ============================================================
// Tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webgpu_config_default() {
        let cfg = WebGpuConfig::default();
        assert!(cfg.prefer_gpu);
        assert!(cfg.fallback_to_cpu);
        assert_eq!(cfg.tile_size, 16);
        assert_eq!(cfg.workgroup_size_x, 16);
        assert_eq!(cfg.workgroup_size_y, 16);
        assert_eq!(cfg.workgroup_size_z, 1);
        assert!(cfg.enable_validation);
    }

    #[test]
    fn test_gpu_buffer_zeros() {
        let buf = GpuBuffer::zeros(8, GpuBufferUsage::Storage);
        assert_eq!(buf.size, 8);
        assert!(buf.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_webgpu_backend_default() {
        let b = WebGpuBackend::default();
        assert_eq!(b, WebGpuBackend::Cpu);
    }

    #[test]
    fn test_gpu_buffer_usage_default() {
        let u = GpuBufferUsage::default();
        assert_eq!(u, GpuBufferUsage::Storage);
    }

    #[test]
    fn test_gpu_buffer_descriptor_default() {
        let d = GpuBufferDescriptor::default();
        assert_eq!(d.size, 0);
        assert_eq!(d.usage, GpuBufferUsage::Storage);
        assert!(d.label.is_none());
    }

    #[test]
    fn test_compute_pipeline_descriptor_default() {
        let d = ComputePipelineDescriptor::default();
        assert_eq!(d.entry_point, "main");
        assert_eq!(d.bind_group_count, 1);
        assert!(d.shader_source.is_empty());
    }

    #[test]
    fn test_gpu_error_display_device_not_available() {
        let e = GpuError::DeviceNotAvailable;
        let s = e.to_string();
        assert!(s.contains("not available"));
    }

    #[test]
    fn test_gpu_error_display_buffer_too_large() {
        let e = GpuError::BufferTooLarge {
            requested: 1024,
            limit: 512,
        };
        let s = e.to_string();
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
    }

    #[test]
    fn test_gpu_error_display_shader_compile() {
        let e = GpuError::ShaderCompile("unexpected token".to_string());
        let s = e.to_string();
        assert!(s.contains("unexpected token"));
    }

    #[test]
    fn test_gpu_error_display_execution() {
        let e = GpuError::Execution("dispatch failed".to_string());
        let s = e.to_string();
        assert!(s.contains("dispatch failed"));
    }

    #[test]
    fn test_gpu_buffer_with_data() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let buf = GpuBuffer::with_data(data.clone(), GpuBufferUsage::Staging);
        assert_eq!(buf.usage, GpuBufferUsage::Staging);
        assert_eq!(buf.data, data);
        assert_eq!(buf.size, 3);
    }
}
