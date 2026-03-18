//! Cross-device memory management for efficient data transfer between CPU and GPU
//!
//! This module provides utilities for managing memory across different devices
//! (CPU, GPU, TPU) with efficient data transfer and synchronization. It includes:
//!
//! - Cross-device memory transfer with automatic format conversion
//! - Memory pools for efficient allocation
//! - Smart caching for frequently accessed data
//! - Efficient pinned memory for faster CPU-GPU transfers
//! - Asynchronous data transfer with event-based synchronization

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
#[cfg(feature = "gpu")]
use crate::gpu::{GpuBackend, GpuBuffer, GpuContext, GpuDataType};
use ::ndarray::{Array, ArrayBase, Dimension, IxDyn, RawData};
use std::any::TypeId;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Device types supported by the cross-device memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU (host) memory
    Cpu,
    /// Discrete GPU memory
    Gpu(GpuBackend),
    /// TPU (Tensor Processing Unit) memory
    Tpu,
}

impl DeviceType {
    /// Check if the device is available on the current system
    pub fn is_available(&self) -> bool {
        match self {
            DeviceType::Cpu => true,
            DeviceType::Gpu(backend) => backend.is_available(),
            DeviceType::Tpu => false, // TPU support not yet implemented
        }
    }

    /// Get the name of the device
    pub fn name(&self) -> String {
        match self {
            DeviceType::Cpu => "CPU".to_string(),
            DeviceType::Gpu(backend) => format!("GPU ({backend})"),
            DeviceType::Tpu => "TPU".to_string(),
        }
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::Gpu(backend) => write!(f, "GPU ({backend})"),
            DeviceType::Tpu => write!(f, "TPU"),
        }
    }
}

/// Memory transfer direction between devices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to device transfer (e.g., CPU to GPU)
    HostToDevice,
    /// Device to host transfer (e.g., GPU to CPU)
    DeviceToHost,
    /// Device to device transfer (e.g., GPU to TPU)
    DeviceToDevice,
}

/// Transfer mode for cross-device operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMode {
    /// Synchronous transfer (blocks until complete)
    Synchronous,
    /// Asynchronous transfer (returns immediately, track with events)
    Asynchronous,
}

/// Memory layout for device buffers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Row-major layout (C-style)
    RowMajor,
    /// Column-major layout (Fortran-style)
    ColumnMajor,
    /// Strided layout (custom strides)
    Strided,
}

/// Options for cross-device memory transfers
#[derive(Debug, Clone)]
pub struct TransferOptions {
    /// Transfer mode (synchronous or asynchronous)
    pub mode: TransferMode,
    /// Memory layout for the transfer
    pub layout: MemoryLayout,
    /// Whether to use pinned memory for the transfer
    pub use_pinned_memory: bool,
    /// Whether to enable streaming transfers for large buffers
    pub enable_streaming: bool,
    /// Stream ID for asynchronous transfers
    pub stream_id: Option<usize>,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            mode: TransferMode::Synchronous,
            layout: MemoryLayout::RowMajor,
            use_pinned_memory: true,
            enable_streaming: true,
            stream_id: None,
        }
    }
}

/// Builder for transfer options
#[derive(Debug, Clone)]
pub struct TransferOptionsBuilder {
    options: TransferOptions,
}

impl TransferOptionsBuilder {
    /// Create a new transfer options builder with default values
    pub fn new() -> Self {
        Self {
            options: TransferOptions::default(),
        }
    }

    /// Set the transfer mode
    pub const fn mode(mut self, mode: TransferMode) -> Self {
        self.options.mode = mode;
        self
    }

    /// Set the memory layout
    pub const fn layout(mut self, layout: MemoryLayout) -> Self {
        self.options.layout = layout;
        self
    }

    /// Set whether to use pinned memory
    pub const fn memory(mut self, use_pinnedmemory: bool) -> Self {
        self.options.use_pinned_memory = use_pinnedmemory;
        self
    }

    /// Set whether to enable streaming transfers
    pub const fn streaming(mut self, enablestreaming: bool) -> Self {
        self.options.enable_streaming = enablestreaming;
        self
    }

    /// Set the stream ID for asynchronous transfers
    pub const fn with_stream_id(mut self, streamid: Option<usize>) -> Self {
        self.options.stream_id = streamid;
        self
    }

    /// Build the transfer options
    pub fn build(self) -> TransferOptions {
        self.options
    }
}

impl Default for TransferOptionsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache key for the device memory cache
#[derive(Debug, Clone, PartialEq, Eq)]
struct CacheKey {
    /// Data identifier (usually the memory address of the host array)
    data_id: usize,
    /// Device type
    device: DeviceType,
    /// Element type ID
    type_id: TypeId,
    /// Size in elements
    size: usize,
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data_id.hash(state);
        self.device.hash(state);
        std::any::TypeId::of::<i32>().hash(state);
        self.size.hash(state);
    }
}

/// Event for tracking asynchronous operations
#[derive(Debug)]
pub struct TransferEvent {
    /// Device associated with the event
    #[allow(dead_code)]
    device: DeviceType,
    /// Internal event handle (implementation-specific)
    #[allow(dead_code)]
    handle: Arc<Mutex<Box<dyn std::any::Any + Send + Sync>>>,
    /// Whether the event has been completed
    completed: Arc<std::sync::atomic::AtomicBool>,
}

impl TransferEvent {
    /// Create a new transfer event
    #[allow(dead_code)]
    fn device(devicetype: DeviceType, handle: Box<dyn std::any::Any + Send + Sync>) -> Self {
        Self {
            device: devicetype,
            handle: Arc::new(Mutex::new(handle)),
            completed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Wait for the event to complete
    pub fn wait(&self) {
        // In a real implementation, this would block until the event is complete
        // For now, just set the completed flag for demonstration
        self.completed
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if the event has completed
    pub fn is_complete(&self) -> bool {
        self.completed.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Cache entry for the device memory cache
struct CacheEntry<T: GpuDataType> {
    /// Buffer on the device
    buffer: DeviceBuffer<T>,
    /// Size in elements
    size: usize,
    /// Last access time
    last_access: std::time::Instant,
    /// Whether the buffer is dirty (modified on device)
    #[allow(dead_code)]
    dirty: bool,
}

/// Device memory manager for cross-device operations
pub struct DeviceMemoryManager {
    /// GPU context for accessing GPU functionality
    gpu_context: Option<GpuContext>,
    /// Cache of device buffers
    cache: Mutex<HashMap<CacheKey, Box<dyn std::any::Any + Send + Sync>>>,
    /// Maximum cache size in bytes
    max_cache_size: usize,
    /// Current cache size in bytes
    current_cache_size: std::sync::atomic::AtomicUsize,
    /// Whether the caching is enabled
    enable_caching: bool,
}

impl std::fmt::Debug for DeviceMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMemoryManager")
            .field("gpu_context", &"<gpu_context>")
            .field("cache", &"<cache>")
            .field("max_cache_size", &self.max_cache_size)
            .field(
                "current_cache_size",
                &self
                    .current_cache_size
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("enable_caching", &self.enable_caching)
            .finish()
    }
}

impl DeviceMemoryManager {
    /// Create a new device memory manager
    pub fn new(max_cachesize: usize) -> Result<Self, CoreError> {
        // Try to create a GPU context if a GPU is available
        let gpu_context = match GpuBackend::preferred() {
            backend if backend.is_available() => GpuContext::new(backend).ok(),
            _ => None,
        };

        Ok(Self {
            gpu_context,
            cache: Mutex::new(HashMap::new()),
            max_cache_size: max_cachesize,
            current_cache_size: std::sync::atomic::AtomicUsize::new(0),
            enable_caching: true,
        })
    }

    /// Check if a device type is available
    pub fn is_device_available(&self, device: DeviceType) -> bool {
        match device {
            DeviceType::Cpu => true,
            DeviceType::Gpu(_) => self.gpu_context.is_some(),
            DeviceType::Tpu => false, // TPU not yet supported
        }
    }

    /// Get a list of available devices
    pub fn available_devices(&self) -> Vec<DeviceType> {
        let mut devices = vec![DeviceType::Cpu];

        if let Some(ref context) = self.gpu_context {
            devices.push(DeviceType::Gpu(context.backend()));
        }

        devices
    }

    /// Transfer data from host to device
    pub fn transfer_to_device<T, S, D>(
        &self,
        array: &ArrayBase<S, D>,
        device: DeviceType,
        options: Option<TransferOptions>,
    ) -> CoreResult<DeviceArray<T, D>>
    where
        T: GpuDataType,
        S: RawData<Elem = T> + crate::ndarray::Data,
        D: Dimension,
    {
        let options = options.unwrap_or_default();

        // Check if the device is available
        if !self.is_device_available(device) {
            return Err(CoreError::DeviceError(
                ErrorContext::new(format!("Device {device} is not available"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // For CPU, just create a view of the array
        if device == DeviceType::Cpu {
            return Ok(DeviceArray::new_cpu(array.to_owned()));
        }

        // For GPU, create a GPU buffer
        if let DeviceType::Gpu(backend) = device {
            if let Some(ref context) = self.gpu_context {
                if context.backend() != backend {
                    return Err(CoreError::DeviceError(
                        ErrorContext::new(format!(
                            "GPU backend mismatch: requested {}, available {}",
                            backend,
                            context.backend()
                        ))
                        .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }

                // Create a flat view of the array data
                let flat_data = array.as_slice().ok_or_else(|| {
                    CoreError::DeviceError(
                        ErrorContext::new("Array is not contiguous".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;

                // Check if we have a cached buffer for this array
                let data_id = flat_data.as_ptr() as usize;
                let key = CacheKey {
                    data_id,
                    device,
                    type_id: TypeId::of::<T>(),
                    size: flat_data.len(),
                };

                let buffer = if self.enable_caching {
                    let mut cache = self.cache.lock().expect("Operation failed");
                    if let Some(entry) = cache.get_mut(&key) {
                        // We found a cached entry, cast it to the correct type
                        if let Some(entry) = entry.downcast_mut::<CacheEntry<T>>() {
                            // Update the last access time
                            entry.last_access = std::time::Instant::now();
                            entry.buffer.clone()
                        } else {
                            // This should never happen if our caching logic is correct
                            return Err(CoreError::DeviceError(
                                ErrorContext::new("Cache entry type mismatch".to_string())
                                    .with_location(ErrorLocation::new(file!(), line!())),
                            ));
                        }
                    } else {
                        // No cached entry, create a new buffer
                        let gpubuffer = context.create_buffer_from_slice(flat_data);
                        let buffer = DeviceBuffer::new_gpu(gpubuffer);

                        // Add to cache
                        let entry = CacheEntry {
                            buffer: buffer.clone(),
                            size: flat_data.len(),
                            last_access: std::time::Instant::now(),
                            dirty: false,
                        };

                        let buffersize = std::mem::size_of_val(flat_data);
                        self.current_cache_size
                            .fetch_add(buffersize, std::sync::atomic::Ordering::SeqCst);

                        // If we're over the cache size limit, evict old entries
                        self.evict_cache_entries_if_needed();

                        cache.insert(key, Box::new(entry));
                        buffer
                    }
                } else {
                    // Caching is disabled, just create a new buffer
                    let gpubuffer = context.create_buffer_from_slice(flat_data);
                    DeviceBuffer::new_gpu(gpubuffer)
                };

                return Ok(DeviceArray {
                    buffer,
                    shape: array.raw_dim(),
                    device: DeviceType::Gpu(crate::gpu::GpuBackend::preferred()),
                    phantom: PhantomData,
                });
            }
        }

        Err(CoreError::DeviceError(
            ErrorContext::new(format!("{device}"))
                .with_location(ErrorLocation::new(file!(), line!())),
        ))
    }

    /// Transfer data from device to host
    pub fn transfer_to_host<T, D>(
        &self,
        devicearray: &DeviceArray<T, D>,
        options: Option<TransferOptions>,
    ) -> CoreResult<Array<T, D>>
    where
        T: GpuDataType,
        D: Dimension,
    {
        let options = options.unwrap_or_default();

        // For CPU arrays, just clone the data
        if devicearray.device == DeviceType::Cpu {
            if let Some(cpuarray) = devicearray.buffer.get_cpuarray() {
                let reshaped = cpuarray
                    .clone()
                    .to_shape(devicearray.shape.clone())
                    .map_err(|e| CoreError::ShapeError(ErrorContext::new(e.to_string())))?
                    .to_owned();
                return Ok(reshaped);
            }
        }

        // For GPU arrays, copy the data back to the host
        if let DeviceType::Gpu(_) = devicearray.device {
            if let Some(gpubuffer) = devicearray.buffer.get_gpubuffer() {
                let size = devicearray.size();
                let mut data = vec![unsafe { std::mem::zeroed() }; size];

                // Copy data from GPU to host
                let _ = gpubuffer.copy_to_host(&mut data);

                // Reshape the data to match the original array shape
                return Array::from_shape_vec(devicearray.shape.clone(), data).map_err(|e| {
                    CoreError::DeviceError(
                        ErrorContext::new(format!("{e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                });
            }
        }

        Err(CoreError::DeviceError(
            ErrorContext::new(format!(
                "Unsupported device type for transfer to host: {}",
                devicearray.device
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ))
    }

    /// Transfer data between devices
    pub fn transfer_between_devices<T, D>(
        &self,
        devicearray: &DeviceArray<T, D>,
        target_device: DeviceType,
        options: Option<TransferOptions>,
    ) -> CoreResult<DeviceArray<T, D>>
    where
        T: GpuDataType,
        D: Dimension,
    {
        let options = options.unwrap_or_default();

        // If the source and target devices are the same, just clone the array
        if devicearray.device == target_device {
            return Ok(devicearray.clone());
        }

        // For transfers to CPU, use transfer_to_host
        if target_device == DeviceType::Cpu {
            let hostarray = self.transfer_to_host(devicearray, Some(options))?;
            return Ok(DeviceArray::new_cpu(hostarray));
        }

        // For transfers from CPU to another device, use transfer_to_device
        if devicearray.device == DeviceType::Cpu {
            if let Some(cpuarray) = devicearray.buffer.get_cpuarray() {
                // Reshape the CPU array to match the expected dimension type
                let cpu_clone = cpuarray.clone();
                let reshaped = cpu_clone
                    .to_shape(devicearray.shape.clone())
                    .map_err(|e| CoreError::ShapeError(ErrorContext::new(e.to_string())))?;
                return self.transfer_to_device(&reshaped.to_owned(), target_device, Some(options));
            }
        }

        // For transfers between GPUs (or future TPU support)
        // In a real implementation, we would use peer-to-peer transfers if available,
        // or copy through host memory if not

        // For now, we'll transfer through host memory
        let hostarray = self.transfer_to_host(devicearray, Some(options.clone()))?;
        self.transfer_to_device(&hostarray, target_device, Some(options))
    }

    /// Evict cache entries if the total size exceeds the limit
    fn evict_cache_entries_if_needed(&self) {
        let current_size = self
            .current_cache_size
            .load(std::sync::atomic::Ordering::SeqCst);
        if current_size <= self.max_cache_size {
            return;
        }

        let mut cache = self.cache.lock().expect("Operation failed");

        // Collect keys with their access times to avoid borrow conflicts
        let mut key_times: Vec<_> = cache
            .iter()
            .map(|(key, value)| {
                let access_time = match value.downcast_ref::<CacheEntry<f32>>() {
                    Some(entry) => entry.last_access,
                    None => match value.downcast_ref::<CacheEntry<f64>>() {
                        Some(entry) => entry.last_access,
                        None => match value.downcast_ref::<CacheEntry<i32>>() {
                            Some(entry) => entry.last_access,
                            None => match value.downcast_ref::<CacheEntry<u32>>() {
                                Some(entry) => entry.last_access,
                                None => std::time::Instant::now(), // Fallback, shouldn't happen
                            },
                        },
                    },
                };
                (key.clone(), access_time)
            })
            .collect();

        // Sort by access time (oldest first)
        key_times.sort_by_key(|a| a.1);

        // Remove entries until we're under the limit
        let mut removed_size = 0;
        let target_size = current_size - self.max_cache_size / 2; // Remove enough to get below half the limit

        for key_ in key_times {
            let entry = cache.remove(&key_.0).expect("Operation failed");

            // Calculate the size of the entry based on its type
            let entry_size = match entry.downcast_ref::<CacheEntry<f32>>() {
                Some(entry) => entry.size * std::mem::size_of::<f32>(),
                None => match entry.downcast_ref::<CacheEntry<f64>>() {
                    Some(entry) => entry.size * std::mem::size_of::<f64>(),
                    None => match entry.downcast_ref::<CacheEntry<i32>>() {
                        Some(entry) => entry.size * std::mem::size_of::<i32>(),
                        None => match entry.downcast_ref::<CacheEntry<u32>>() {
                            Some(entry) => entry.size * std::mem::size_of::<u32>(),
                            None => 0, // Fallback, shouldn't happen
                        },
                    },
                },
            };

            removed_size += entry_size;

            if removed_size >= target_size {
                break;
            }
        }

        // Update the current cache size
        self.current_cache_size
            .fetch_sub(removed_size, std::sync::atomic::Ordering::SeqCst);
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().expect("Operation failed");
        cache.clear();
        self.current_cache_size
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }

    /// Execute a kernel on a device array
    pub fn execute_kernel<T, D>(
        &self,
        devicearray: &DeviceArray<T, D>,
        kernel_name: &str,
        params: HashMap<String, KernelParam>,
    ) -> CoreResult<()>
    where
        T: GpuDataType,
        D: Dimension,
    {
        // Only GPU devices support kernel execution
        if let DeviceType::Gpu(_) = devicearray.device {
            if let Some(ref context) = self.gpu_context {
                // Get the kernel
                let kernel = context
                    .get_kernel(kernel_name)
                    .map_err(|e| CoreError::ComputationError(ErrorContext::new(e.to_string())))?;

                // Set the input buffer parameter
                if let Some(gpubuffer) = devicearray.buffer.get_gpubuffer() {
                    kernel.set_buffer("input", gpubuffer);
                }

                // Set other parameters
                for (name, param) in params {
                    match param {
                        KernelParam::Buffer(buffer) => {
                            if let Some(gpubuffer) = buffer.get_gpubuffer() {
                                kernel.set_buffer(&name, gpubuffer);
                            }
                        }
                        KernelParam::U32(value) => kernel.set_u32(&name, value),
                        KernelParam::I32(value) => kernel.set_i32(&name, value),
                        KernelParam::F32(value) => kernel.set_f32(&name, value),
                        KernelParam::F64(value) => kernel.set_f64(&name, value),
                    }
                }

                // Compute dispatch dimensions
                let total_elements = devicearray.size();
                let work_group_size = 256; // A common CUDA/OpenCL work group size
                let num_groups = total_elements.div_ceil(work_group_size);

                // Dispatch the kernel
                kernel.dispatch([num_groups as u32, 1, 1]);

                return Ok(());
            }
        }

        Err(CoreError::DeviceError(
            ErrorContext::new(format!(
                "Unsupported device type for kernel execution: {}",
                devicearray.device
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ))
    }
}

/// Kernel parameter for GPU execution
#[derive(Debug, Clone)]
pub enum KernelParam {
    /// Buffer parameter
    Buffer(DeviceBuffer<f32>), // Note: In a real implementation, this would be generic
    /// U32 parameter
    U32(u32),
    /// I32 parameter
    I32(i32),
    /// F32 parameter
    F32(f32),
    /// F64 parameter
    F64(f64),
}

/// Buffer location (CPU or GPU)
#[derive(Clone)]
enum BufferLocation<T: GpuDataType> {
    /// CPU buffer
    Cpu(Arc<Array<T, IxDyn>>),
    /// GPU buffer
    Gpu(Arc<GpuBuffer<T>>),
}

impl<T> std::fmt::Debug for BufferLocation<T>
where
    T: GpuDataType + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferLocation::Cpu(_) => write!(f, "Cpu(Array)"),
            BufferLocation::Gpu(_) => write!(f, "Gpu(GpuBuffer)"),
        }
    }
}

/// Buffer for cross-device operations
#[derive(Debug, Clone)]
pub struct DeviceBuffer<T: GpuDataType> {
    /// Buffer data (CPU or GPU)
    location: BufferLocation<T>,
}

impl<T: GpuDataType> DeviceBuffer<T> {
    /// Create a new CPU buffer
    fn new_cpu<D: Dimension>(array: Array<T, D>) -> Self {
        Self {
            location: BufferLocation::Cpu(Arc::new(array.into_dyn())),
        }
    }

    /// Create a new GPU buffer
    fn new_gpu(buffer: GpuBuffer<T>) -> Self {
        Self {
            location: BufferLocation::Gpu(Arc::new(buffer)),
        }
    }

    /// Get the CPU array if available
    fn get_cpuarray(&self) -> Option<&Array<T, IxDyn>> {
        match self.location {
            BufferLocation::Cpu(ref array) => Some(array),
            _ => None,
        }
    }

    /// Get the GPU buffer if available
    fn get_gpubuffer(&self) -> Option<&GpuBuffer<T>> {
        match self.location {
            BufferLocation::Gpu(ref buffer) => Some(buffer),
            _ => None,
        }
    }

    /// Get the size of the buffer in elements
    fn size(&self) -> usize {
        match self.location {
            BufferLocation::Cpu(ref array) => array.len(),
            BufferLocation::Gpu(ref buffer) => buffer.len(),
        }
    }
}

/// Array residing on a specific device (CPU, GPU, TPU)
#[derive(Debug, Clone)]
pub struct DeviceArray<T: GpuDataType, D: Dimension> {
    /// Buffer containing the array data
    buffer: DeviceBuffer<T>,
    /// Shape of the array
    shape: D,
    /// Device where the array resides
    device: DeviceType,
    /// Phantom data for the element type
    phantom: PhantomData<T>,
}

impl<T: GpuDataType, D: Dimension> DeviceArray<T, D> {
    /// Create a new CPU array
    fn new_cpu<S: RawData<Elem = T> + crate::ndarray::Data>(array: ArrayBase<S, D>) -> Self {
        Self {
            buffer: DeviceBuffer::new_cpu(array.to_owned()),
            shape: array.raw_dim(),
            device: DeviceType::Cpu,
            phantom: PhantomData,
        }
    }

    /// Get the device where the array resides
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Get the shape of the array
    pub const fn shape(&self) -> &D {
        &self.shape
    }

    /// Get the size of the array in elements
    pub fn size(&self) -> usize {
        self.buffer.size()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Check if the array is on the CPU
    pub fn is_on_cpu(&self) -> bool {
        self.device == DeviceType::Cpu
    }

    /// Check if the array is on a GPU
    pub fn is_on_gpu(&self) -> bool {
        matches!(self.device, DeviceType::Gpu(_))
    }

    /// Get a reference to the underlying CPU array if available
    pub fn as_cpuarray(&self) -> Option<&Array<T, IxDyn>> {
        self.buffer.get_cpuarray()
    }

    /// Get a reference to the underlying GPU buffer if available
    pub fn as_gpubuffer(&self) -> Option<&GpuBuffer<T>> {
        self.buffer.get_gpubuffer()
    }
}

/// Stream for asynchronous operations
pub struct DeviceStream {
    /// Device associated with the stream
    #[allow(dead_code)]
    device: DeviceType,
    /// Internal stream handle (implementation-specific)
    #[allow(dead_code)]
    handle: Arc<Mutex<Box<dyn std::any::Any + Send + Sync>>>,
}

impl DeviceStream {
    /// Create a new device stream
    pub fn new(device: DeviceType) -> CoreResult<Self> {
        // In a real implementation, we would create a stream for the _device
        // For now, just create a dummy stream
        Ok(Self {
            device,
            handle: Arc::new(Mutex::new(Box::new(()))),
        })
    }

    /// Synchronize the stream
    pub fn synchronize(&self) {
        // In a real implementation, this would wait for all operations to complete
    }
}

/// Memory pool for efficient allocation on a device
pub struct DeviceMemoryPool {
    /// Device associated with the pool
    device: DeviceType,
    /// List of free buffers by size
    freebuffers: Mutex<HashMap<usize, Vec<Box<dyn std::any::Any + Send + Sync>>>>,
    /// Maximum pool size in bytes
    max_poolsize: usize,
    /// Current pool size in bytes
    current_poolsize: std::sync::atomic::AtomicUsize,
}

impl std::fmt::Debug for DeviceMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMemoryPool")
            .field("device", &self.device)
            .field("freebuffers", &"<freebuffers>")
            .field("max_poolsize", &self.max_poolsize)
            .field(
                "current_poolsize",
                &self
                    .current_poolsize
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

impl DeviceMemoryPool {
    /// Create a new device memory pool
    pub fn new(device: DeviceType, max_poolsize: usize) -> Self {
        Self {
            device,
            freebuffers: Mutex::new(HashMap::new()),
            max_poolsize,
            current_poolsize: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Allocate a buffer of the given size
    pub fn allocate<T: GpuDataType + num_traits::Zero>(
        &self,
        size: usize,
    ) -> CoreResult<DeviceBuffer<T>> {
        // Check if we have a free buffer of the right size
        let mut freebuffers = self.freebuffers.lock().expect("Operation failed");
        if let Some(buffers) = freebuffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                // We found a free buffer, cast it to the correct type
                if let Ok(buffer) = buffer.downcast::<DeviceBuffer<T>>() {
                    return Ok(*buffer);
                }
            }
        }

        // No free buffer, allocate a new one
        match self.device {
            DeviceType::Cpu => {
                // Allocate CPU memory
                let array = Array::<T, crate::ndarray::IxDyn>::zeros(IxDyn(&[size]));
                Ok(DeviceBuffer::new_cpu(array))
            }
            DeviceType::Gpu(_) => {
                // Allocate GPU memory
                Err(CoreError::ImplementationError(
                    ErrorContext::new("GPU memory allocation not implemented".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ))
            }
            DeviceType::Tpu => {
                // TPU not yet supported
                Err(CoreError::DeviceError(
                    ErrorContext::new("TPU not supported".to_string())
                        .with_location(ErrorLocation::new(file!(), line!())),
                ))
            }
        }
    }

    /// Free a buffer (return it to the pool)
    pub fn free<T: GpuDataType>(&self, buffer: DeviceBuffer<T>) {
        let size = buffer.size();
        let buffersize = size * std::mem::size_of::<T>();

        // Check if adding this buffer would exceed the pool size
        let current_size = self
            .current_poolsize
            .load(std::sync::atomic::Ordering::SeqCst);
        if current_size + buffersize > self.max_poolsize {
            // Pool is full, just let the buffer be dropped
            return;
        }

        // Add the buffer to the pool
        let mut freebuffers = self.freebuffers.lock().expect("Operation failed");
        freebuffers.entry(size).or_default().push(Box::new(buffer));

        // Update the pool size
        self.current_poolsize
            .fetch_add(buffersize, std::sync::atomic::Ordering::SeqCst);
    }

    /// Clear the pool
    pub fn clear(&self) {
        let mut freebuffers = self.freebuffers.lock().expect("Operation failed");
        freebuffers.clear();
        self.current_poolsize
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Cross-device array operations
impl<T: GpuDataType, D: Dimension> DeviceArray<T, D> {
    /// Map the array elements using a function
    pub fn map<F>(&self, f: F, manager: &DeviceMemoryManager) -> CoreResult<DeviceArray<T, D>>
    where
        F: Fn(T) -> T + Send + Sync,
        D: Clone,
    {
        // For CPU arrays, use ndarray's map function
        if self.is_on_cpu() {
            if let Some(cpuarray) = self.as_cpuarray() {
                let mapped = cpuarray.map(|&x| f(x));
                return Ok(DeviceArray {
                    buffer: DeviceBuffer::new_cpu(mapped),
                    shape: self.shape.clone(),
                    device: DeviceType::Cpu,
                    phantom: PhantomData,
                });
            }
        }

        // For GPU arrays, transfer to host, map, and transfer back
        // In a real implementation, we would use a GPU kernel
        let hostarray = manager.transfer_to_host(self, None)?;
        let mapped = hostarray.map(|&x| f(x));
        manager.transfer_to_device(&mapped, self.device, None)
    }

    /// Reduce the array using a binary operation
    pub fn reduce<F>(&self, f: F, manager: &DeviceMemoryManager) -> CoreResult<T>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: Copy,
    {
        // For CPU arrays, use ndarray's fold function
        if self.is_on_cpu() {
            if let Some(cpuarray) = self.as_cpuarray() {
                if cpuarray.is_empty() {
                    return Err(CoreError::ValueError(
                        ErrorContext::new("Cannot reduce empty array".to_string())
                            .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }

                let first = cpuarray[0];
                let result = cpuarray.iter().skip(1).fold(first, |acc, &x| f(acc, x));
                return Ok(result);
            }
        }

        // For GPU arrays, transfer to host and reduce
        // In a real implementation, we would use a GPU reduction kernel
        let hostarray = manager.transfer_to_host(self, None)?;
        if hostarray.is_empty() {
            return Err(CoreError::ValueError(
                ErrorContext::new("Cannot reduce empty array".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let first = *hostarray.iter().next().expect("Operation failed");
        let result = hostarray.iter().skip(1).fold(first, |acc, &x| f(acc, x));
        Ok(result)
    }
}

/// Cross-device manager for handling data transfers and operations
#[derive(Debug)]
pub struct CrossDeviceManager {
    /// Memory managers for each device
    memory_managers: HashMap<DeviceType, DeviceMemoryManager>,
    /// Memory pools for each device
    memory_pools: HashMap<DeviceType, DeviceMemoryPool>,
    /// Active data transfers
    active_transfers: Mutex<Vec<TransferEvent>>,
    /// Enable caching
    #[allow(dead_code)]
    enable_caching: bool,
    /// Maximum cache size in bytes
    #[allow(dead_code)]
    max_cache_size: usize,
}

impl CrossDeviceManager {
    /// Create a new cross-device manager
    pub fn new(max_cachesize: usize) -> CoreResult<Self> {
        let mut memory_managers = HashMap::new();
        let mut memory_pools = HashMap::new();

        // Create CPU memory manager and pool
        let cpu_manager = DeviceMemoryManager::new(max_cachesize)?;
        memory_managers.insert(DeviceType::Cpu, cpu_manager);
        memory_pools.insert(
            DeviceType::Cpu,
            DeviceMemoryPool::new(DeviceType::Cpu, max_cachesize),
        );

        // Try to create GPU memory manager and pool
        let gpu_backend = GpuBackend::preferred();
        if gpu_backend.is_available() {
            let gpu_device = DeviceType::Gpu(gpu_backend);
            let gpu_manager = DeviceMemoryManager::new(max_cachesize)?;
            memory_managers.insert(gpu_device, gpu_manager);
            memory_pools.insert(gpu_device, DeviceMemoryPool::new(gpu_device, max_cachesize));
        }

        Ok(Self {
            memory_managers,
            memory_pools,
            active_transfers: Mutex::new(Vec::new()),
            enable_caching: true,
            max_cache_size: max_cachesize,
        })
    }

    /// Get a list of available devices
    pub fn available_devices(&self) -> Vec<DeviceType> {
        self.memory_managers.keys().cloned().collect()
    }

    /// Check if a device is available
    pub fn is_device_available(&self, device: DeviceType) -> bool {
        self.memory_managers.contains_key(&device)
    }

    /// Transfer data to a device
    pub fn to_device<T, S, D>(
        &self,
        array: &ArrayBase<S, D>,
        device: DeviceType,
        options: Option<TransferOptions>,
    ) -> CoreResult<DeviceArray<T, D>>
    where
        T: GpuDataType,
        S: RawData<Elem = T> + crate::ndarray::Data,
        D: Dimension,
    {
        // Check if the device is available
        if !self.is_device_available(device) {
            return Err(CoreError::DeviceError(
                ErrorContext::new(format!("Device {device} is not available"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Get the memory manager for the device
        let manager = self.memory_managers.get(&device).expect("Operation failed");
        manager.transfer_to_device(array, device, options)
    }

    /// Transfer data from a device to the host
    pub fn to_host<T, D>(
        &self,
        devicearray: &DeviceArray<T, D>,
        options: Option<TransferOptions>,
    ) -> CoreResult<Array<T, D>>
    where
        T: GpuDataType,
        D: Dimension,
    {
        // Get the memory manager for the device
        let manager = self
            .memory_managers
            .get(&devicearray.device)
            .ok_or_else(|| {
                CoreError::DeviceError(
                    ErrorContext::new(format!("Device {} is not available", devicearray.device))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        manager.transfer_to_host(devicearray, options)
    }

    /// Transfer data between devices
    pub fn transfer<T, D>(
        &self,
        devicearray: &DeviceArray<T, D>,
        target_device: DeviceType,
        options: Option<TransferOptions>,
    ) -> CoreResult<DeviceArray<T, D>>
    where
        T: GpuDataType,
        D: Dimension,
    {
        // Check if the target _device is available
        if !self.is_device_available(target_device) {
            return Err(CoreError::DeviceError(
                ErrorContext::new(format!("Device {target_device} is not available"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Get the memory manager for the source _device
        let manager = self
            .memory_managers
            .get(&devicearray.device)
            .ok_or_else(|| {
                CoreError::DeviceError(
                    ErrorContext::new(format!("Device {} is not available", devicearray.device))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        manager.transfer_between_devices(devicearray, target_device, options)
    }

    /// Execute a kernel on a device array
    pub fn execute_kernel<T, D>(
        &self,
        devicearray: &DeviceArray<T, D>,
        kernel_name: &str,
        params: HashMap<String, KernelParam>,
    ) -> CoreResult<()>
    where
        T: GpuDataType,
        D: Dimension,
    {
        // Get the memory manager for the device
        let manager = self
            .memory_managers
            .get(&devicearray.device)
            .ok_or_else(|| {
                CoreError::DeviceError(
                    ErrorContext::new(format!("Device {} is not available", devicearray.device))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        manager.execute_kernel(devicearray, kernel_name, params)
    }

    /// Allocate memory on a device
    pub fn allocate<T: GpuDataType + num_traits::Zero>(
        &self,
        size: usize,
        device: DeviceType,
    ) -> CoreResult<DeviceBuffer<T>> {
        // Check if the device is available
        if !self.is_device_available(device) {
            return Err(CoreError::DeviceError(
                ErrorContext::new(format!("Device {device} is not available"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Get the memory pool for the device
        let pool = self.memory_pools.get(&device).expect("Operation failed");
        pool.allocate(size)
    }

    /// Free memory on a device
    pub fn free<T: GpuDataType>(&self, buffer: DeviceBuffer<T>, device: DeviceType) {
        // Check if the device is available
        if !self.is_device_available(device) {
            return;
        }

        // Get the memory pool for the device
        let pool = self.memory_pools.get(&device).expect("Operation failed");
        pool.free(buffer);
    }

    /// Clear all caches and pools
    pub fn clear(&self) {
        // Clear all memory managers
        for manager in self.memory_managers.values() {
            manager.clear_cache();
        }

        // Clear all memory pools
        for pool in self.memory_pools.values() {
            pool.clear();
        }

        // Clear active transfers
        let mut active_transfers = self.active_transfers.lock().expect("Operation failed");
        active_transfers.clear();
    }

    /// Wait for all active transfers to complete
    pub fn synchronize(&self) {
        let mut active_transfers = self.active_transfers.lock().expect("Operation failed");
        for event in active_transfers.drain(..) {
            event.wait();
        }
    }
}

/// Create a cross-device manager with default settings
#[allow(dead_code)]
pub fn create_cross_device_manager() -> CoreResult<CrossDeviceManager> {
    CrossDeviceManager::new(1024 * 1024 * 1024) // 1 GB cache by default
}

/// Extension trait for arrays to simplify device transfers
pub trait ToDevice<T, D>
where
    T: GpuDataType,
    D: Dimension,
{
    /// Transfer the array to a device
    fn to_device(
        &self,
        device: DeviceType,
        manager: &CrossDeviceManager,
    ) -> CoreResult<DeviceArray<T, D>>;
}

impl<T, S, D> ToDevice<T, D> for ArrayBase<S, D>
where
    T: GpuDataType,
    S: RawData<Elem = T> + crate::ndarray::Data,
    D: Dimension,
{
    fn to_device(
        &self,
        device: DeviceType,
        manager: &CrossDeviceManager,
    ) -> CoreResult<DeviceArray<T, D>> {
        manager.to_device(self, device, None)
    }
}

/// Extension trait for device arrays to simplify host transfers
pub trait ToHost<T, D>
where
    T: GpuDataType,
    D: Dimension,
{
    /// Transfer the device array to the host
    fn to_host(&self, manager: &CrossDeviceManager) -> CoreResult<Array<T, D>>;
}

impl<T, D> ToHost<T, D> for DeviceArray<T, D>
where
    T: GpuDataType,
    D: Dimension,
{
    fn to_host(&self, manager: &CrossDeviceManager) -> CoreResult<Array<T, D>> {
        manager.to_host(self, None)
    }
}

// Convenience functions

/// Create a device array on the CPU
#[allow(dead_code)]
pub fn create_cpuarray<T, S, D>(array: &ArrayBase<S, D>) -> DeviceArray<T, D>
where
    T: GpuDataType,
    S: RawData<Elem = T> + crate::ndarray::Data,
    D: Dimension,
{
    DeviceArray::new_cpu(array.to_owned())
}

/// Create a device array on the GPU
#[allow(dead_code)]
pub fn create_gpuarray<T, S, D>(
    array: &ArrayBase<S, D>,
    manager: &CrossDeviceManager,
) -> CoreResult<DeviceArray<T, D>>
where
    T: GpuDataType,
    S: RawData<Elem = T> + crate::ndarray::Data,
    D: Dimension,
{
    // Find the first available GPU
    for device in manager.available_devices() {
        if let DeviceType::Gpu(_) = device {
            return manager.to_device(array, device, None);
        }
    }

    Err(CoreError::DeviceError(
        ErrorContext::new("No GPU device available".to_string())
            .with_location(ErrorLocation::new(file!(), line!())),
    ))
}

/// Transfer an array to the best available device
#[allow(dead_code)]
pub fn to_best_device<T, S, D>(
    array: &ArrayBase<S, D>,
    manager: &CrossDeviceManager,
) -> CoreResult<DeviceArray<T, D>>
where
    T: GpuDataType,
    S: RawData<Elem = T> + crate::ndarray::Data,
    D: Dimension,
{
    // Try to find a GPU first
    for device in manager.available_devices() {
        if let DeviceType::Gpu(_) = device {
            return manager.to_device(array, device, None);
        }
    }

    // Fall back to CPU
    Ok(create_cpuarray(array))
}
