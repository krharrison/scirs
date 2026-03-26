//! GPU acceleration for sparse matrix operations
//!
//! This module provides GPU acceleration support for sparse matrix operations
//! across multiple backends including CUDA, OpenCL, Metal, ROCm, and WGPU.

pub mod construction;
pub mod cuda;
pub mod metal;
pub mod opencl;
pub mod solvers;
pub mod spmv;
pub mod vulkan;

// Re-export common types and traits
#[cfg(feature = "gpu")]
pub use scirs2_core::gpu::{
    GpuBackend, GpuBuffer, GpuContext, GpuDataType, GpuDevice, GpuError, GpuKernelHandle,
};

// Fallback types when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
pub use crate::gpu_ops::{
    GpuBackend, GpuBuffer, GpuDataType, GpuDevice, GpuError, GpuKernelHandle,
};

// Re-export backend-specific modules
pub use cuda::{CudaMemoryManager, CudaOptimizationLevel, CudaSpMatVec};
pub use metal::{MetalDeviceInfo, MetalMemoryManager, MetalOptimizationLevel, MetalSpMatVec};
pub use opencl::{
    OpenCLMemoryManager, OpenCLOptimizationLevel, OpenCLPlatformInfo, OpenCLSpMatVec,
};
pub use vulkan::{
    VulkanDeviceInfo, VulkanDeviceType, VulkanMemoryManager, VulkanOptimizationLevel,
    VulkanSpMatVec,
};

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, SparseElement};
use std::fmt::Debug;

// GpuDataType is already available from the pub use statements above

/// Unified GPU sparse matrix operations interface
pub struct GpuSpMatVec {
    backend: GpuBackend,
    cuda_handler: Option<CudaSpMatVec>,
    opencl_handler: Option<OpenCLSpMatVec>,
    metal_handler: Option<MetalSpMatVec>,
    vulkan_handler: Option<VulkanSpMatVec>,
}

impl GpuSpMatVec {
    /// Create a new GPU sparse matrix handler with automatic backend detection
    pub fn new() -> SparseResult<Self> {
        let backend = Self::detect_best_backend();

        let mut handler = Self {
            backend,
            cuda_handler: None,
            opencl_handler: None,
            metal_handler: None,
            vulkan_handler: None,
        };

        // Initialize the appropriate backend
        handler.initialize_backend()?;

        Ok(handler)
    }

    /// Create a new GPU sparse matrix handler with specified backend
    pub fn with_backend(backend: GpuBackend) -> SparseResult<Self> {
        let mut handler = Self {
            backend,
            cuda_handler: None,
            opencl_handler: None,
            metal_handler: None,
            vulkan_handler: None,
        };

        handler.initialize_backend()?;

        Ok(handler)
    }

    /// Initialize the selected backend
    fn initialize_backend(&mut self) -> SparseResult<()> {
        match self.backend {
            GpuBackend::Cuda => {
                self.cuda_handler = Some(CudaSpMatVec::new()?);
            }
            GpuBackend::OpenCL => {
                self.opencl_handler = Some(OpenCLSpMatVec::new()?);
            }
            GpuBackend::Metal => {
                self.metal_handler = Some(MetalSpMatVec::new()?);
            }
            GpuBackend::Wgpu => {
                self.vulkan_handler = Some(VulkanSpMatVec::new()?);
            }
            #[cfg(not(feature = "gpu"))]
            GpuBackend::Vulkan => {
                self.vulkan_handler = Some(VulkanSpMatVec::new()?);
            }
            GpuBackend::Cpu => {
                // CPU fallback - no initialization needed
            }
            GpuBackend::Rocm => {
                // For ROCm, fall back to CPU for now
                self.backend = GpuBackend::Cpu;
            }
        }

        Ok(())
    }

    /// Detect the best available GPU backend
    fn detect_best_backend() -> GpuBackend {
        // Priority order: Metal (on macOS), CUDA, Vulkan, OpenCL, CPU
        #[cfg(target_os = "macos")]
        {
            if Self::is_metal_available() {
                return GpuBackend::Metal;
            }
        }

        if Self::is_cuda_available() {
            return GpuBackend::Cuda;
        }

        if Self::is_vulkan_available() {
            return GpuBackend::Wgpu;
        }

        if Self::is_opencl_available() {
            return GpuBackend::OpenCL;
        }

        GpuBackend::Cpu
    }

    /// Check if CUDA is available
    fn is_cuda_available() -> bool {
        // In a real implementation, this would check for CUDA runtime
        #[cfg(feature = "gpu")]
        {
            // Simplified detection
            std::env::var("CUDA_PATH").is_ok() || std::path::Path::new("/usr/local/cuda").exists()
        }
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// Check if OpenCL is available
    fn is_opencl_available() -> bool {
        // In a real implementation, this would check for OpenCL runtime
        #[cfg(feature = "gpu")]
        {
            // Simplified detection - assume available on most systems
            true
        }
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// Check if Metal is available (macOS only)
    fn is_metal_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            // Metal is available on all modern macOS systems
            true
        }
        #[cfg(not(target_os = "macos"))]
        false
    }

    /// Check if Vulkan is available
    fn is_vulkan_available() -> bool {
        #[cfg(feature = "gpu")]
        {
            // Check for Vulkan SDK or runtime
            std::env::var("VULKAN_SDK").is_ok()
                || std::path::Path::new("/usr/share/vulkan").exists()
                || std::path::Path::new("/usr/local/share/vulkan").exists()
        }
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// Execute sparse matrix-vector multiplication on GPU
    pub fn spmv<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        device: Option<&GpuDevice>,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + SparseElement + Debug + Copy + GpuDataType + std::iter::Sum,
    {
        match self.backend {
            GpuBackend::Cuda => {
                if let Some(ref handler) = self.cuda_handler {
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(device) = device {
                            handler.execute_spmv(matrix, vector, device)
                        } else {
                            // Fall back to CPU when no device is provided
                            matrix.dot_vector(vector)
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    handler.execute_spmv_cpu(matrix, vector)
                } else {
                    Err(SparseError::ComputationError(
                        "CUDA handler not initialized".to_string(),
                    ))
                }
            }
            GpuBackend::OpenCL => {
                if let Some(ref handler) = self.opencl_handler {
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(device) = device {
                            handler.execute_spmv(matrix, vector, device)
                        } else {
                            // Fall back to CPU when no device is provided
                            matrix.dot_vector(vector)
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    handler.execute_spmv_cpu(matrix, vector)
                } else {
                    Err(SparseError::ComputationError(
                        "OpenCL handler not initialized".to_string(),
                    ))
                }
            }
            GpuBackend::Metal => {
                if let Some(ref handler) = self.metal_handler {
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(device) = device {
                            handler.execute_spmv(matrix, vector, device)
                        } else {
                            // Fall back to CPU when no device is provided
                            matrix.dot_vector(vector)
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    handler.execute_spmv_cpu(matrix, vector)
                } else {
                    Err(SparseError::ComputationError(
                        "Metal handler not initialized".to_string(),
                    ))
                }
            }
            GpuBackend::Wgpu => {
                if let Some(ref handler) = self.vulkan_handler {
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(device) = device {
                            handler.execute_spmv(matrix, vector, device)
                        } else {
                            // Fall back to CPU when no device is provided
                            matrix.dot_vector(vector)
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    handler.execute_spmv_cpu(matrix, vector)
                } else {
                    Err(SparseError::ComputationError(
                        "Vulkan handler not initialized".to_string(),
                    ))
                }
            }
            #[cfg(not(feature = "gpu"))]
            GpuBackend::Vulkan => {
                if let Some(ref handler) = self.vulkan_handler {
                    handler.execute_spmv_cpu(matrix, vector)
                } else {
                    Err(SparseError::ComputationError(
                        "Vulkan handler not initialized".to_string(),
                    ))
                }
            }
            GpuBackend::Cpu => {
                // CPU fallback
                matrix.dot_vector(vector)
            }
            GpuBackend::Rocm => {
                // Unsupported backend, fall back to CPU
                matrix.dot_vector(vector)
            }
        }
    }

    /// Execute optimized sparse matrix-vector multiplication
    pub fn spmv_optimized<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        device: Option<&GpuDevice>,
        optimization_hint: OptimizationHint,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + SparseElement + Debug + Copy + GpuDataType + std::iter::Sum,
    {
        match self.backend {
            GpuBackend::Cuda => {
                if let Some(ref handler) = self.cuda_handler {
                    let cuda_level = optimization_hint.to_cuda_level();
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(device) = device {
                            handler.execute_optimized_spmv(matrix, vector, device, cuda_level)
                        } else {
                            return Err(SparseError::ComputationError(
                                "GPU device required for CUDA operations".to_string(),
                            ));
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    handler.execute_spmv_cpu(matrix, vector)
                } else {
                    Err(SparseError::ComputationError(
                        "CUDA handler not initialized".to_string(),
                    ))
                }
            }
            GpuBackend::OpenCL => {
                if let Some(ref handler) = self.opencl_handler {
                    let opencl_level = optimization_hint.to_opencl_level();
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(device) = device {
                            handler.execute_optimized_spmv(matrix, vector, device, opencl_level)
                        } else {
                            return Err(SparseError::ComputationError(
                                "GPU device required for OpenCL operations".to_string(),
                            ));
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    handler.execute_spmv_cpu(matrix, vector)
                } else {
                    Err(SparseError::ComputationError(
                        "OpenCL handler not initialized".to_string(),
                    ))
                }
            }
            GpuBackend::Metal => {
                if let Some(ref handler) = self.metal_handler {
                    let metal_level = optimization_hint.to_metal_level();
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(device) = device {
                            handler.execute_optimized_spmv(matrix, vector, device, metal_level)
                        } else {
                            return Err(SparseError::ComputationError(
                                "GPU device required for Metal operations".to_string(),
                            ));
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    handler.execute_spmv_cpu(matrix, vector)
                } else {
                    Err(SparseError::ComputationError(
                        "Metal handler not initialized".to_string(),
                    ))
                }
            }
            GpuBackend::Wgpu => {
                if let Some(ref handler) = self.vulkan_handler {
                    let vulkan_level = optimization_hint.to_vulkan_level();
                    #[cfg(feature = "gpu")]
                    {
                        if let Some(device) = device {
                            handler.execute_optimized_spmv(matrix, vector, device, vulkan_level)
                        } else {
                            return Err(SparseError::ComputationError(
                                "GPU device required for Vulkan operations".to_string(),
                            ));
                        }
                    }
                    #[cfg(not(feature = "gpu"))]
                    handler.execute_spmv_cpu(matrix, vector)
                } else {
                    Err(SparseError::ComputationError(
                        "Vulkan handler not initialized".to_string(),
                    ))
                }
            }
            #[cfg(not(feature = "gpu"))]
            GpuBackend::Vulkan => {
                if let Some(ref handler) = self.vulkan_handler {
                    let vulkan_level = optimization_hint.to_vulkan_level();
                    handler.execute_spmv_cpu(matrix, vector)
                } else {
                    Err(SparseError::ComputationError(
                        "Vulkan handler not initialized".to_string(),
                    ))
                }
            }
            GpuBackend::Cpu | GpuBackend::Rocm => {
                // Fall back to basic implementation
                self.spmv(matrix, vector, device)
            }
        }
    }

    /// Get the current backend
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        !matches!(self.backend, GpuBackend::Cpu)
    }

    /// Get backend-specific information
    pub fn get_backend_info(&self) -> BackendInfo {
        match self.backend {
            GpuBackend::Cuda => BackendInfo {
                name: "CUDA".to_string(),
                version: "Unknown".to_string(),
                device_count: 1, // Simplified
                supports_double_precision: true,
                max_memory_mb: 8192, // 8GB default
            },
            GpuBackend::OpenCL => BackendInfo {
                name: "OpenCL".to_string(),
                version: "Unknown".to_string(),
                device_count: 1,
                supports_double_precision: true,
                max_memory_mb: 4096, // 4GB default
            },
            GpuBackend::Metal => BackendInfo {
                name: "Metal".to_string(),
                version: "Unknown".to_string(),
                device_count: 1,
                supports_double_precision: false, // Metal has limited f64 support
                max_memory_mb: if MetalDeviceInfo::detect().is_apple_silicon {
                    16384
                } else {
                    8192
                },
            },
            GpuBackend::Wgpu => BackendInfo {
                name: "Vulkan".to_string(),
                version: "Unknown".to_string(),
                device_count: 1,
                supports_double_precision: true,
                max_memory_mb: 8192, // 8GB default
            },
            #[cfg(not(feature = "gpu"))]
            GpuBackend::Vulkan => BackendInfo {
                name: "Vulkan".to_string(),
                version: "Unknown".to_string(),
                device_count: 1,
                supports_double_precision: true,
                max_memory_mb: 8192, // 8GB default
            },
            GpuBackend::Cpu | GpuBackend::Rocm => BackendInfo {
                name: "CPU".to_string(),
                version: "Fallback".to_string(),
                device_count: 0,
                supports_double_precision: true,
                max_memory_mb: 0,
            },
        }
    }
}

impl Default for GpuSpMatVec {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            backend: GpuBackend::Cpu,
            cuda_handler: None,
            opencl_handler: None,
            metal_handler: None,
            vulkan_handler: None,
        })
    }
}

/// Cross-platform optimization hints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationHint {
    /// Basic optimization level
    Basic,
    /// Balanced optimization (default)
    #[default]
    Balanced,
    /// Maximum performance optimization
    Maximum,
    /// Memory-optimized implementation
    MemoryOptimized,
}

impl OptimizationHint {
    /// Convert to CUDA optimization level
    pub fn to_cuda_level(self) -> CudaOptimizationLevel {
        match self {
            OptimizationHint::Basic => CudaOptimizationLevel::Basic,
            OptimizationHint::Balanced => CudaOptimizationLevel::Vectorized,
            OptimizationHint::Maximum => CudaOptimizationLevel::WarpLevel,
            OptimizationHint::MemoryOptimized => CudaOptimizationLevel::Basic,
        }
    }

    /// Convert to OpenCL optimization level
    pub fn to_opencl_level(self) -> OpenCLOptimizationLevel {
        match self {
            OptimizationHint::Basic => OpenCLOptimizationLevel::Basic,
            OptimizationHint::Balanced => OpenCLOptimizationLevel::Workgroup,
            OptimizationHint::Maximum => OpenCLOptimizationLevel::Vectorized,
            OptimizationHint::MemoryOptimized => OpenCLOptimizationLevel::Workgroup,
        }
    }

    /// Convert to Metal optimization level
    pub fn to_metal_level(self) -> MetalOptimizationLevel {
        match self {
            OptimizationHint::Basic => MetalOptimizationLevel::Basic,
            OptimizationHint::Balanced => MetalOptimizationLevel::SimdGroup,
            OptimizationHint::Maximum => MetalOptimizationLevel::AppleSilicon,
            OptimizationHint::MemoryOptimized => MetalOptimizationLevel::AppleSilicon,
        }
    }

    /// Convert to Vulkan optimization level
    pub fn to_vulkan_level(self) -> VulkanOptimizationLevel {
        match self {
            OptimizationHint::Basic => VulkanOptimizationLevel::Basic,
            OptimizationHint::Balanced => VulkanOptimizationLevel::ComputeShader,
            OptimizationHint::Maximum => VulkanOptimizationLevel::Maximum,
            OptimizationHint::MemoryOptimized => VulkanOptimizationLevel::Subgroup,
        }
    }
}

/// GPU backend information
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub name: String,
    pub version: String,
    pub device_count: usize,
    pub supports_double_precision: bool,
    pub max_memory_mb: usize,
}

/// Convenient functions for common operations
pub mod convenience {
    use super::*;

    /// Execute sparse matrix-vector multiplication with automatic GPU detection
    pub fn gpu_spmv<T>(matrix: &CsrArray<T>, vector: &ArrayView1<T>) -> SparseResult<Array1<T>>
    where
        T: Float + SparseElement + Debug + Copy + GpuDataType + std::iter::Sum,
    {
        let gpu_handler = GpuSpMatVec::new()?;
        gpu_handler.spmv(matrix, vector, None)
    }

    /// Execute optimized sparse matrix-vector multiplication
    pub fn gpu_spmv_optimized<T>(
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        optimization: OptimizationHint,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + SparseElement + Debug + Copy + GpuDataType + std::iter::Sum,
    {
        let gpu_handler = GpuSpMatVec::new()?;
        gpu_handler.spmv_optimized(matrix, vector, None, optimization)
    }

    /// Get information about available GPU backends
    pub fn available_backends() -> Vec<GpuBackend> {
        let mut backends = Vec::new();

        if GpuSpMatVec::is_cuda_available() {
            backends.push(GpuBackend::Cuda);
        }

        if GpuSpMatVec::is_vulkan_available() {
            backends.push(GpuBackend::Wgpu);
        }

        if GpuSpMatVec::is_opencl_available() {
            backends.push(GpuBackend::OpenCL);
        }

        if GpuSpMatVec::is_metal_available() {
            backends.push(GpuBackend::Metal);
        }

        backends.push(GpuBackend::Cpu); // Always available

        backends
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_gpu_spmv_creation() {
        let gpu_spmv = GpuSpMatVec::new();
        assert!(gpu_spmv.is_ok());
    }

    #[test]
    fn test_backend_detection() {
        let backend = GpuSpMatVec::detect_best_backend();

        // Should return a valid backend
        match backend {
            GpuBackend::Cuda
            | GpuBackend::OpenCL
            | GpuBackend::Metal
            | GpuBackend::Cpu
            | GpuBackend::Wgpu
            | GpuBackend::Rocm => (),
            #[cfg(not(feature = "gpu"))]
            GpuBackend::Vulkan => (),
        }
    }

    #[test]
    fn test_optimization_hint_conversions() {
        let hint = OptimizationHint::Maximum;

        let cuda_level = hint.to_cuda_level();
        let opencl_level = hint.to_opencl_level();
        let metal_level = hint.to_metal_level();

        assert_eq!(cuda_level, CudaOptimizationLevel::WarpLevel);
        assert_eq!(opencl_level, OpenCLOptimizationLevel::Vectorized);
        assert_eq!(metal_level, MetalOptimizationLevel::AppleSilicon);
    }

    #[test]
    fn test_backend_info() {
        let gpu_spmv = GpuSpMatVec::new().expect("Operation failed");
        let info = gpu_spmv.get_backend_info();

        assert!(!info.name.is_empty());
        assert!(!info.version.is_empty());
    }

    #[test]
    fn test_convenience_functions() {
        let backends = convenience::available_backends();
        assert!(!backends.is_empty());
        assert!(backends.contains(&GpuBackend::Cpu)); // CPU should always be available
    }

    #[test]
    fn test_is_gpu_available() {
        let gpu_spmv = GpuSpMatVec::new().expect("Operation failed");

        // Should not panic - either true or false is valid
        let _available = gpu_spmv.is_gpu_available();
    }

    #[test]
    fn test_optimization_hint_default() {
        assert_eq!(OptimizationHint::default(), OptimizationHint::Balanced);
    }
}
