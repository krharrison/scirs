//! CPU-side GPU buffer abstraction for scirs2-core.
//!
//! Provides a clean API layer that simulates GPU buffer operations in pure-Rust
//! CPU-only environments. Can be backed by actual GPU later.

use std::sync::{Arc, Mutex};

use crate::error::{CoreError, CoreResult};

// ─────────────────────────────────────────────────────────────────────────────
// Device / DType
// ─────────────────────────────────────────────────────────────────────────────

/// Device type for buffer placement.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU (host) memory.
    Cpu,
    /// GPU device.  In pure-Rust builds this is simulated on the CPU.
    Gpu {
        /// Zero-based GPU device index.
        device_id: u32,
    },
    /// WebGPU device.
    WebGpu,
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Cpu
    }
}

/// Scalar data type stored in a [`GpuBuffer`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// Unsigned byte.
    U8,
}

impl DType {
    /// Size in bytes of a single element.
    pub fn bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// GPU buffer abstraction.
///
/// In pure-Rust builds the backing store is a heap-allocated `Vec<u8>`.
/// The API is intentionally designed so a real GPU backend can replace the
/// inner storage without touching user code.
#[derive(Clone)]
pub struct GpuBuffer {
    data: Arc<Mutex<Vec<u8>>>,
    shape: Vec<usize>,
    dtype: DType,
    device: DeviceType,
    label: Option<String>,
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .field("label", &self.label)
            .finish()
    }
}

impl GpuBuffer {
    // ── constructors ──────────────────────────────────────────────────────

    /// Allocate a zero-filled buffer.
    pub fn zeros(shape: &[usize], dtype: DType, device: DeviceType) -> Self {
        let n = shape.iter().product::<usize>() * dtype.bytes();
        GpuBuffer {
            data: Arc::new(Mutex::new(vec![0u8; n])),
            shape: shape.to_vec(),
            dtype,
            device,
            label: None,
        }
    }

    /// Create a buffer from a `Vec<f32>`.
    ///
    /// The shape is inferred as `[data.len()]`.
    pub fn from_vec_f32(data: Vec<f32>, shape: &[usize]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        GpuBuffer {
            data: Arc::new(Mutex::new(raw)),
            shape: shape.to_vec(),
            dtype: DType::F32,
            device: DeviceType::Cpu,
            label: None,
        }
    }

    /// Create a buffer from a `Vec<f64>`.
    ///
    /// The shape is inferred as `[data.len()]`.
    pub fn from_vec_f64(data: Vec<f64>, shape: &[usize]) -> Self {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        GpuBuffer {
            data: Arc::new(Mutex::new(raw)),
            shape: shape.to_vec(),
            dtype: DType::F64,
            device: DeviceType::Cpu,
            label: None,
        }
    }

    // ── metadata ──────────────────────────────────────────────────────────

    /// Logical shape of the buffer.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Device this buffer lives on.
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Total number of scalar elements.
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total number of bytes.
    pub fn n_bytes(&self) -> usize {
        self.n_elements() * self.dtype.bytes()
    }

    /// Attach a human-readable label (builder style).
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_owned());
        self
    }

    /// Return the label if set.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    // ── host → device upload ──────────────────────────────────────────────

    /// Copy `f32` data from a host slice into the buffer.
    pub fn upload_f32(&self, src: &[f32]) -> CoreResult<()> {
        if self.dtype != DType::F32 {
            return Err(CoreError::InvalidArgument(
                crate::error::ErrorContext::new("upload_f32: buffer dtype is not F32"),
            ));
        }
        if src.len() != self.n_elements() {
            return Err(CoreError::ShapeError(crate::error::ErrorContext::new(
                format!(
                    "upload_f32: src len {} != buffer elements {}",
                    src.len(),
                    self.n_elements()
                ),
            )));
        }
        let raw: Vec<u8> = src.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut guard = self
            .data
            .lock()
            .map_err(|_| CoreError::ComputationError(crate::error::ErrorContext::new("mutex poisoned")))?;
        *guard = raw;
        Ok(())
    }

    /// Copy `f64` data from a host slice into the buffer.
    pub fn upload_f64(&self, src: &[f64]) -> CoreResult<()> {
        if self.dtype != DType::F64 {
            return Err(CoreError::InvalidArgument(
                crate::error::ErrorContext::new("upload_f64: buffer dtype is not F64"),
            ));
        }
        if src.len() != self.n_elements() {
            return Err(CoreError::ShapeError(crate::error::ErrorContext::new(
                format!(
                    "upload_f64: src len {} != buffer elements {}",
                    src.len(),
                    self.n_elements()
                ),
            )));
        }
        let raw: Vec<u8> = src.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut guard = self
            .data
            .lock()
            .map_err(|_| CoreError::ComputationError(crate::error::ErrorContext::new("mutex poisoned")))?;
        *guard = raw;
        Ok(())
    }

    // ── device → host download ────────────────────────────────────────────

    /// Download the buffer as `Vec<f32>`.
    pub fn to_vec_f32(&self) -> CoreResult<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(CoreError::InvalidArgument(
                crate::error::ErrorContext::new("to_vec_f32: buffer dtype is not F32"),
            ));
        }
        let guard = self
            .data
            .lock()
            .map_err(|_| CoreError::ComputationError(crate::error::ErrorContext::new("mutex poisoned")))?;
        let out: Vec<f32> = guard
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(out)
    }

    /// Download the buffer as `Vec<f64>`.
    pub fn to_vec_f64(&self) -> CoreResult<Vec<f64>> {
        if self.dtype != DType::F64 {
            return Err(CoreError::InvalidArgument(
                crate::error::ErrorContext::new("to_vec_f64: buffer dtype is not F64"),
            ));
        }
        let guard = self
            .data
            .lock()
            .map_err(|_| CoreError::ComputationError(crate::error::ErrorContext::new("mutex poisoned")))?;
        let out: Vec<f64> = guard
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        Ok(out)
    }

    // ── compute operations (CPU simulation) ───────────────────────────────

    /// Element-wise addition.  Returns a new buffer; both inputs are unchanged.
    pub fn add(&self, other: &GpuBuffer) -> CoreResult<GpuBuffer> {
        if self.dtype != other.dtype {
            return Err(CoreError::InvalidArgument(
                crate::error::ErrorContext::new("add: dtype mismatch"),
            ));
        }
        if self.shape != other.shape {
            return Err(CoreError::ShapeError(crate::error::ErrorContext::new(
                "add: shape mismatch",
            )));
        }
        match self.dtype {
            DType::F32 => {
                let a = self.to_vec_f32()?;
                let b = other.to_vec_f32()?;
                let c: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                Ok(GpuBuffer::from_vec_f32(c, &self.shape))
            }
            DType::F64 => {
                let a = self.to_vec_f64()?;
                let b = other.to_vec_f64()?;
                let c: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
                Ok(GpuBuffer::from_vec_f64(c, &self.shape))
            }
            _ => Err(CoreError::InvalidArgument(
                crate::error::ErrorContext::new("add: unsupported dtype for element-wise add"),
            )),
        }
    }

    /// Scalar multiplication.  Returns a new buffer.
    pub fn scale(&self, scalar: f64) -> CoreResult<GpuBuffer> {
        match self.dtype {
            DType::F32 => {
                let v = self.to_vec_f32()?;
                let out: Vec<f32> = v.iter().map(|x| (*x as f64 * scalar) as f32).collect();
                Ok(GpuBuffer::from_vec_f32(out, &self.shape))
            }
            DType::F64 => {
                let v = self.to_vec_f64()?;
                let out: Vec<f64> = v.iter().map(|x| x * scalar).collect();
                Ok(GpuBuffer::from_vec_f64(out, &self.shape))
            }
            _ => Err(CoreError::InvalidArgument(
                crate::error::ErrorContext::new("scale: unsupported dtype"),
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputeDispatch
// ─────────────────────────────────────────────────────────────────────────────

/// Compute shader dispatch descriptor.
///
/// In pure-Rust builds the "shader" is executed as a closure on the CPU.
pub struct ComputeDispatch {
    /// Work-group size in X, Y, Z.
    pub workgroup_size: [u32; 3],
    /// Number of work-groups in X, Y, Z.
    pub n_workgroups: [u32; 3],
}

impl ComputeDispatch {
    /// Create a dispatch sized to cover `total_threads` threads.
    ///
    /// `workgroup_size` specifies the per-axis group sizes; the number of
    /// work-groups is rounded up so that `n_workgroups * workgroup_size >=
    /// total_threads` along the X axis.  Y and Z are set to 1.
    pub fn new(total_threads: usize, workgroup_size: [u32; 3]) -> Self {
        let wx = workgroup_size[0].max(1) as usize;
        let nx = total_threads.div_ceil(wx) as u32;
        ComputeDispatch {
            workgroup_size,
            n_workgroups: [nx, 1, 1],
        }
    }

    /// Total number of threads launched.
    pub fn total_threads(&self) -> u64 {
        let ws: u64 = self.workgroup_size.iter().map(|&x| x as u64).product();
        let ng: u64 = self.n_workgroups.iter().map(|&x| x as u64).product();
        ws * ng
    }

    /// Execute a closure as if it were a compute shader.
    ///
    /// The closure receives the global thread ID `(x, y, z)`.  In this CPU
    /// simulation the closure is called sequentially for every global thread.
    pub fn execute<F>(&self, kernel: F) -> CoreResult<()>
    where
        F: Fn(u32, u32, u32) + Sync + Send,
    {
        let gx = self.workgroup_size[0] * self.n_workgroups[0];
        let gy = self.workgroup_size[1] * self.n_workgroups[1];
        let gz = self.workgroup_size[2] * self.n_workgroups[2];
        for z in 0..gz {
            for y in 0..gy {
                for x in 0..gx {
                    kernel(x, y, z);
                }
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation_zeros() {
        let buf = GpuBuffer::zeros(&[4, 4], DType::F32, DeviceType::Cpu);
        assert_eq!(buf.shape(), &[4, 4]);
        assert_eq!(buf.dtype(), DType::F32);
        assert_eq!(buf.device(), DeviceType::Cpu);
        assert_eq!(buf.n_elements(), 16);
        assert_eq!(buf.n_bytes(), 64);
    }

    #[test]
    fn test_shape_queries() {
        let buf = GpuBuffer::zeros(&[2, 3, 5], DType::F64, DeviceType::Gpu { device_id: 0 });
        assert_eq!(buf.n_elements(), 30);
        assert_eq!(buf.n_bytes(), 240);
        assert_eq!(buf.device(), DeviceType::Gpu { device_id: 0 });
    }

    #[test]
    fn test_upload_download_roundtrip_f32() {
        let buf = GpuBuffer::zeros(&[4], DType::F32, DeviceType::Cpu);
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        buf.upload_f32(&src).expect("upload_f32 failed");
        let dst = buf.to_vec_f32().expect("to_vec_f32 failed");
        assert_eq!(dst, src);
    }

    #[test]
    fn test_upload_download_roundtrip_f64() {
        let buf = GpuBuffer::zeros(&[3], DType::F64, DeviceType::Cpu);
        let src = vec![1.5f64, -2.5, 3.14159];
        buf.upload_f64(&src).expect("upload_f64 failed");
        let dst = buf.to_vec_f64().expect("to_vec_f64 failed");
        for (a, b) in dst.iter().zip(src.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_add_operation() {
        let a = GpuBuffer::from_vec_f32(vec![1.0, 2.0, 3.0], &[3]);
        let b = GpuBuffer::from_vec_f32(vec![4.0, 5.0, 6.0], &[3]);
        let c = a.add(&b).expect("add failed");
        let result = c.to_vec_f32().expect("to_vec_f32 failed");
        assert_eq!(result, vec![5.0f32, 7.0, 9.0]);
    }

    #[test]
    fn test_scale_operation() {
        let buf = GpuBuffer::from_vec_f64(vec![1.0, 2.0, 3.0], &[3]);
        let scaled = buf.scale(2.5).expect("scale failed");
        let result = scaled.to_vec_f64().expect("to_vec_f64 failed");
        assert!((result[0] - 2.5).abs() < 1e-12);
        assert!((result[1] - 5.0).abs() < 1e-12);
        assert!((result[2] - 7.5).abs() < 1e-12);
    }

    #[test]
    fn test_label() {
        let buf = GpuBuffer::zeros(&[8], DType::F32, DeviceType::Cpu)
            .with_label("my_weights");
        assert_eq!(buf.label(), Some("my_weights"));
    }

    #[test]
    fn test_compute_dispatch() {
        use std::sync::{Arc, Mutex};
        let counter = Arc::new(Mutex::new(0u32));
        let counter_clone = Arc::clone(&counter);
        let dispatch = ComputeDispatch::new(16, [8, 1, 1]);
        dispatch
            .execute(move |_x, _y, _z| {
                let mut c = counter_clone.lock().expect("lock failed");
                *c += 1;
            })
            .expect("execute failed");
        let total = *counter.lock().expect("lock failed");
        // n_workgroups[0] = ceil(16/8)=2, so total threads = 2*8 = 16
        assert_eq!(total, 16);
    }

    #[test]
    fn test_from_vec_shape_and_content() {
        let data = vec![10.0f32, 20.0, 30.0, 40.0];
        let buf = GpuBuffer::from_vec_f32(data.clone(), &[2, 2]);
        assert_eq!(buf.shape(), &[2, 2]);
        let out = buf.to_vec_f32().expect("to_vec_f32 failed");
        assert_eq!(out, data);
    }
}
