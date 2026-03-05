//! Async Out-of-Core I/O for SciRS2
//!
//! Provides `AsyncOutOfCoreArray<T>` for working with arrays larger than available memory
//! using asynchronous chunk-based loading with configurable chunk sizes, LRU caching,
//! and streaming iterator interfaces.
//!
//! # Features
//!
//! - Chunk-based loading with configurable chunk sizes
//! - Async loading with tokio integration (feature-gated under `async`)
//! - LRU cache for loaded chunks
//! - Synchronous iterator interface for streaming through chunks
//! - Dirty tracking and write-back for modified chunks
//!
//! # Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────┐
//!  │     AsyncOutOfCoreArray<T>          │
//!  │  ┌──────────┐  ┌────────────────┐  │
//!  │  │ LRU Cache│  │ Chunk Registry │  │
//!  │  └──────────┘  └────────────────┘  │
//!  │         │              │            │
//!  │         ▼              ▼            │
//!  │  ┌──────────────────────────────┐  │
//!  │  │      File Storage Backend    │  │
//!  │  │  (raw binary chunk files)    │  │
//!  │  └──────────────────────────────┘  │
//!  └─────────────────────────────────────┘
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the async out-of-core array.
#[derive(Debug, Clone)]
pub struct AsyncOutOfCoreConfig {
    /// Number of elements per chunk
    pub chunk_size: usize,
    /// Maximum number of chunks to keep in the LRU cache
    pub max_cached_chunks: usize,
    /// Maximum total bytes for the cache
    pub max_cache_bytes: usize,
    /// Enable write-back of dirty chunks on eviction
    pub enable_write_back: bool,
    /// Number of chunks to prefetch ahead (0 = disabled)
    pub prefetch_ahead: usize,
    /// I/O buffer size for file operations
    pub io_buffer_size: usize,
}

impl Default for AsyncOutOfCoreConfig {
    fn default() -> Self {
        Self {
            chunk_size: 65536, // 64K elements per chunk
            max_cached_chunks: 32,
            max_cache_bytes: 256 * 1024 * 1024, // 256 MB
            enable_write_back: true,
            prefetch_ahead: 2,
            io_buffer_size: 64 * 1024, // 64 KB
        }
    }
}

impl AsyncOutOfCoreConfig {
    /// Create a configuration optimized for sequential reading
    pub fn sequential_read() -> Self {
        Self {
            chunk_size: 131072,                 // 128K elements
            max_cached_chunks: 8,               // Few chunks since sequential
            max_cache_bytes: 128 * 1024 * 1024, // 128 MB
            enable_write_back: false,
            prefetch_ahead: 4,          // Aggressive prefetch
            io_buffer_size: 256 * 1024, // 256 KB buffer
        }
    }

    /// Create a configuration optimized for random access
    pub fn random_access() -> Self {
        Self {
            chunk_size: 16384,                  // 16K elements (smaller chunks)
            max_cached_chunks: 128,             // Many chunks
            max_cache_bytes: 512 * 1024 * 1024, // 512 MB
            enable_write_back: true,
            prefetch_ahead: 0, // No prefetch for random
            io_buffer_size: 64 * 1024,
        }
    }

    /// Create a configuration optimized for large datasets
    pub fn large_dataset() -> Self {
        Self {
            chunk_size: 262144, // 256K elements
            max_cached_chunks: 16,
            max_cache_bytes: 1024 * 1024 * 1024, // 1 GB
            enable_write_back: true,
            prefetch_ahead: 2,
            io_buffer_size: 1024 * 1024, // 1 MB buffer
        }
    }
}

// ============================================================================
// LRU Cache
// ============================================================================

/// An LRU cache entry holding chunk data and metadata.
#[derive(Debug)]
struct CacheEntry<T> {
    /// The chunk data
    data: Vec<T>,
    /// Whether the chunk has been modified since loading
    dirty: bool,
    /// Last access time
    last_accessed: Instant,
    /// Number of accesses
    access_count: u64,
}

/// LRU cache for chunk data.
struct LruChunkCache<T> {
    /// Cached entries, keyed by chunk index
    entries: HashMap<usize, CacheEntry<T>>,
    /// Access order for LRU eviction (front = least recent)
    access_order: VecDeque<usize>,
    /// Maximum number of entries
    max_entries: usize,
    /// Maximum total bytes
    max_bytes: usize,
    /// Current total bytes used
    current_bytes: usize,
    /// Element size in bytes
    element_size: usize,
    /// Cache statistics
    hits: u64,
    misses: u64,
}

impl<T: Clone> LruChunkCache<T> {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            max_entries,
            max_bytes,
            current_bytes: 0,
            element_size: std::mem::size_of::<T>(),
            hits: 0,
            misses: 0,
        }
    }

    /// Get a chunk from the cache, updating access order
    fn get(&mut self, chunk_idx: usize) -> Option<&[T]> {
        if self.entries.contains_key(&chunk_idx) {
            self.hits += 1;
            // Move to back (most recently used)
            self.access_order.retain(|&idx| idx != chunk_idx);
            self.access_order.push_back(chunk_idx);
            // Update access metadata
            if let Some(entry) = self.entries.get_mut(&chunk_idx) {
                entry.last_accessed = Instant::now();
                entry.access_count += 1;
            }
            self.entries.get(&chunk_idx).map(|e| e.data.as_slice())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Get a mutable reference to chunk data
    fn get_mut(&mut self, chunk_idx: usize) -> Option<&mut [T]> {
        if self.entries.contains_key(&chunk_idx) {
            self.hits += 1;
            self.access_order.retain(|&idx| idx != chunk_idx);
            self.access_order.push_back(chunk_idx);
            if let Some(entry) = self.entries.get_mut(&chunk_idx) {
                entry.last_accessed = Instant::now();
                entry.access_count += 1;
                entry.dirty = true;
                Some(entry.data.as_mut_slice())
            } else {
                None
            }
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a chunk into the cache, evicting if necessary.
    /// Returns a list of evicted (chunk_idx, data, dirty) tuples.
    fn insert(&mut self, chunk_idx: usize, data: Vec<T>) -> Vec<(usize, Vec<T>, bool)> {
        let entry_bytes = data.len() * self.element_size;
        let mut evicted = Vec::new();

        // Evict until we have room
        while (self.entries.len() >= self.max_entries
            || self.current_bytes + entry_bytes > self.max_bytes)
            && !self.access_order.is_empty()
        {
            if let Some(evict_idx) = self.access_order.pop_front() {
                if let Some(evict_entry) = self.entries.remove(&evict_idx) {
                    let evict_bytes = evict_entry.data.len() * self.element_size;
                    self.current_bytes = self.current_bytes.saturating_sub(evict_bytes);
                    evicted.push((evict_idx, evict_entry.data, evict_entry.dirty));
                }
            }
        }

        // Remove existing entry if present
        if let Some(old_entry) = self.entries.remove(&chunk_idx) {
            let old_bytes = old_entry.data.len() * self.element_size;
            self.current_bytes = self.current_bytes.saturating_sub(old_bytes);
            self.access_order.retain(|&idx| idx != chunk_idx);
        }

        // Insert new entry
        self.current_bytes += entry_bytes;
        self.entries.insert(
            chunk_idx,
            CacheEntry {
                data,
                dirty: false,
                last_accessed: Instant::now(),
                access_count: 1,
            },
        );
        self.access_order.push_back(chunk_idx);

        evicted
    }

    /// Mark a chunk as dirty
    fn mark_dirty(&mut self, chunk_idx: usize) {
        if let Some(entry) = self.entries.get_mut(&chunk_idx) {
            entry.dirty = true;
        }
    }

    /// Get all dirty chunks
    fn dirty_chunks(&self) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|(_, entry)| entry.dirty)
            .map(|(&idx, _)| idx)
            .collect()
    }

    /// Mark a chunk as clean
    fn mark_clean(&mut self, chunk_idx: usize) {
        if let Some(entry) = self.entries.get_mut(&chunk_idx) {
            entry.dirty = false;
        }
    }

    /// Get cache statistics
    fn statistics(&self) -> CacheStatistics {
        let hit_rate = if self.hits + self.misses > 0 {
            self.hits as f64 / (self.hits + self.misses) as f64
        } else {
            0.0
        };
        CacheStatistics {
            cached_chunks: self.entries.len(),
            memory_usage: self.current_bytes,
            max_memory: self.max_bytes,
            hit_rate,
            hits: self.hits,
            misses: self.misses,
            dirty_chunks: self.entries.values().filter(|e| e.dirty).count(),
        }
    }

    /// Clear the cache, returning all dirty entries
    fn clear(&mut self) -> Vec<(usize, Vec<T>, bool)> {
        let mut evicted = Vec::new();
        for (idx, entry) in self.entries.drain() {
            if entry.dirty {
                evicted.push((idx, entry.data, true));
            }
        }
        self.access_order.clear();
        self.current_bytes = 0;
        evicted
    }
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Number of chunks currently cached
    pub cached_chunks: usize,
    /// Current memory usage in bytes
    pub memory_usage: usize,
    /// Maximum memory capacity in bytes
    pub max_memory: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Number of dirty (modified) chunks in cache
    pub dirty_chunks: usize,
}

// ============================================================================
// File header for the out-of-core array
// ============================================================================

/// File header for the async out-of-core array data file.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AsyncOocHeader {
    /// Magic bytes: "SCI2AOOC"
    magic: [u8; 8],
    /// Version
    version: u32,
    /// Element size in bytes
    element_size: u32,
    /// Total number of elements
    total_elements: u64,
    /// Elements per chunk
    chunk_size: u64,
    /// Number of chunks
    num_chunks: u64,
    /// Reserved
    _reserved: [u8; 24],
}

impl AsyncOocHeader {
    const MAGIC: [u8; 8] = *b"SCI2AOOC";
    const VERSION: u32 = 1;
    const HEADER_SIZE: usize = std::mem::size_of::<Self>();

    fn new<T>(total_elements: usize, chunk_size: usize) -> Self {
        let num_chunks = total_elements.div_ceil(chunk_size);
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            element_size: std::mem::size_of::<T>() as u32,
            total_elements: total_elements as u64,
            chunk_size: chunk_size as u64,
            num_chunks: num_chunks as u64,
            _reserved: [0u8; 24],
        }
    }

    fn validate<T>(&self) -> CoreResult<()> {
        if self.magic != Self::MAGIC {
            return Err(CoreError::ValidationError(
                ErrorContext::new("Invalid magic bytes in AsyncOutOfCore file".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        if self.version != Self::VERSION {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Unsupported file version: {} (expected {})",
                    self.version,
                    Self::VERSION
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        if self.element_size != std::mem::size_of::<T>() as u32 {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Element size mismatch: file has {}, expected {} for type {}",
                    self.element_size,
                    std::mem::size_of::<T>(),
                    std::any::type_name::<T>()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    fn from_bytes(bytes: &[u8]) -> CoreResult<Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return Err(CoreError::ValidationError(
                ErrorContext::new("File too small for header".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        // Copy header bytes to ensure alignment
        let mut header = Self {
            magic: [0; 8],
            version: 0,
            element_size: 0,
            total_elements: 0,
            chunk_size: 0,
            num_chunks: 0,
            _reserved: [0; 24],
        };
        let header_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                &mut header as *mut Self as *mut u8,
                std::mem::size_of::<Self>(),
            )
        };
        header_bytes.copy_from_slice(&bytes[..std::mem::size_of::<Self>()]);
        Ok(header)
    }
}

// ============================================================================
// AsyncOutOfCoreArray
// ============================================================================

/// An array that stores data on disk and loads chunks on demand.
///
/// `AsyncOutOfCoreArray<T>` manages arrays too large to fit in memory by
/// transparently loading and caching chunks from a file-backed store.
///
/// # Type Requirements
///
/// `T` must be `Copy + Default + Send + Sync + 'static` to support
/// zero-copy I/O and safe concurrent access.
///
/// # Example (synchronous)
///
/// ```rust,no_run
/// # #[cfg(feature = "memory_efficient")]
/// # {
/// use scirs2_core::memory_efficient::async_out_of_core::{
///     AsyncOutOfCoreArray, AsyncOutOfCoreConfig,
/// };
/// use std::path::Path;
///
/// let config = AsyncOutOfCoreConfig::default();
/// let data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();
/// let path = Path::new("/tmp/large_array.ooc");
///
/// let arr = AsyncOutOfCoreArray::from_slice(&data, path, config)
///     .expect("Failed to create out-of-core array");
///
/// // Access elements (chunk is loaded transparently)
/// let val = arr.get(500_000).expect("Failed to get element");
/// assert!((val - 500_000.0).abs() < 1e-10);
///
/// // Stream through all chunks
/// for chunk in arr.chunk_iter().expect("Failed to create iterator") {
///     let chunk_data = chunk.expect("Failed to load chunk");
///     // Process chunk_data...
/// }
/// # }
/// ```
pub struct AsyncOutOfCoreArray<T: Copy + Default + Send + Sync + 'static> {
    /// Path to the backing file
    file_path: PathBuf,
    /// File handle for I/O
    file: Arc<Mutex<File>>,
    /// LRU cache for chunks
    cache: Arc<RwLock<LruChunkCache<T>>>,
    /// Configuration
    config: AsyncOutOfCoreConfig,
    /// Total number of elements
    total_elements: usize,
    /// Number of chunks
    num_chunks: usize,
    /// Data offset in the file (after header)
    data_offset: usize,
    /// Phantom type marker
    _phantom: PhantomData<T>,
}

impl<T: Copy + Default + Send + Sync + 'static> AsyncOutOfCoreArray<T> {
    /// Create a new out-of-core array from a slice, writing all data to disk.
    pub fn from_slice(data: &[T], path: &Path, config: AsyncOutOfCoreConfig) -> CoreResult<Self> {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("Zero-sized types are not supported".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let total_elements = data.len();
        let chunk_size = config.chunk_size;
        let header = AsyncOocHeader::new::<T>(total_elements, chunk_size);

        // Write header and data
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to create file {}: {e}", path.display()))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        file.write_all(header.to_bytes()).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to write header: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Write data as raw bytes
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        file.write_all(data_bytes).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to write data: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        file.flush().map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to flush: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        let num_chunks = total_elements.div_ceil(chunk_size);

        Ok(Self {
            file_path: path.to_path_buf(),
            file: Arc::new(Mutex::new(file)),
            cache: Arc::new(RwLock::new(LruChunkCache::new(
                config.max_cached_chunks,
                config.max_cache_bytes,
            ))),
            config,
            total_elements,
            num_chunks,
            data_offset: AsyncOocHeader::HEADER_SIZE,
            _phantom: PhantomData,
        })
    }

    /// Open an existing out-of-core array file.
    pub fn open(path: &Path, config: AsyncOutOfCoreConfig) -> CoreResult<Self> {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new("Zero-sized types are not supported".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to open file {}: {e}", path.display()))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        // Read header
        let mut header_buf = vec![0u8; AsyncOocHeader::HEADER_SIZE];
        file.read_exact(&mut header_buf).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to read header: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        let header = AsyncOocHeader::from_bytes(&header_buf)?;
        header.validate::<T>()?;

        let total_elements = header.total_elements as usize;
        let chunk_size = header.chunk_size as usize;
        let num_chunks = header.num_chunks as usize;

        // Override config chunk_size with file's chunk_size for consistency
        let mut effective_config = config;
        effective_config.chunk_size = chunk_size;

        Ok(Self {
            file_path: path.to_path_buf(),
            file: Arc::new(Mutex::new(file)),
            cache: Arc::new(RwLock::new(LruChunkCache::new(
                effective_config.max_cached_chunks,
                effective_config.max_cache_bytes,
            ))),
            config: effective_config,
            total_elements,
            num_chunks,
            data_offset: AsyncOocHeader::HEADER_SIZE,
            _phantom: PhantomData,
        })
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.total_elements
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.total_elements == 0
    }

    /// Get the number of chunks
    pub fn num_chunks(&self) -> usize {
        self.num_chunks
    }

    /// Get the chunk size (elements per chunk)
    pub fn chunk_size(&self) -> usize {
        self.config.chunk_size
    }

    /// Get the file path
    pub fn path(&self) -> &Path {
        &self.file_path
    }

    /// Get cache statistics
    pub fn cache_statistics(&self) -> CoreResult<CacheStatistics> {
        let cache = self.cache.read().map_err(|e| {
            CoreError::MemoryError(
                ErrorContext::new(format!("Failed to read cache: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        Ok(cache.statistics())
    }

    /// Get a single element by global index.
    ///
    /// Loads the corresponding chunk into cache if not already cached.
    pub fn get(&self, index: usize) -> CoreResult<T> {
        if index >= self.total_elements {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new(format!(
                    "Index {} out of bounds for array of length {}",
                    index, self.total_elements
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let chunk_idx = index / self.config.chunk_size;
        let offset_in_chunk = index % self.config.chunk_size;

        let chunk = self.load_chunk(chunk_idx)?;

        if offset_in_chunk >= chunk.len() {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new(format!(
                    "Offset {} out of bounds within chunk of size {}",
                    offset_in_chunk,
                    chunk.len()
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        Ok(chunk[offset_in_chunk])
    }

    /// Set a single element by global index.
    ///
    /// Loads the corresponding chunk, modifies it, and marks it as dirty.
    pub fn set(&self, index: usize, value: T) -> CoreResult<()> {
        if index >= self.total_elements {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new(format!(
                    "Index {} out of bounds for array of length {}",
                    index, self.total_elements
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let chunk_idx = index / self.config.chunk_size;
        let offset_in_chunk = index % self.config.chunk_size;

        // Ensure the chunk is loaded
        self.load_chunk(chunk_idx)?;

        // Get mutable access and modify
        let mut cache = self.cache.write().map_err(|e| {
            CoreError::MemoryError(
                ErrorContext::new(format!("Failed to write-lock cache: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        if let Some(chunk_data) = cache.get_mut(chunk_idx) {
            if offset_in_chunk < chunk_data.len() {
                chunk_data[offset_in_chunk] = value;
                Ok(())
            } else {
                Err(CoreError::InvalidArgument(
                    ErrorContext::new(format!(
                        "Offset {} out of bounds within chunk of size {}",
                        offset_in_chunk,
                        chunk_data.len()
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ))
            }
        } else {
            Err(CoreError::MemoryError(
                ErrorContext::new("Chunk was not in cache after loading".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ))
        }
    }

    /// Load a chunk from disk into the cache if not already cached.
    /// Returns a copy of the chunk data.
    fn load_chunk(&self, chunk_idx: usize) -> CoreResult<Vec<T>> {
        if chunk_idx >= self.num_chunks {
            return Err(CoreError::InvalidArgument(
                ErrorContext::new(format!(
                    "Chunk index {} out of bounds (num_chunks: {})",
                    chunk_idx, self.num_chunks
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        // Check cache first
        {
            let mut cache = self.cache.write().map_err(|e| {
                CoreError::MemoryError(
                    ErrorContext::new(format!("Failed to lock cache: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

            if let Some(data) = cache.get(chunk_idx) {
                return Ok(data.to_vec());
            }
        }

        // Read from file
        let chunk_data = self.read_chunk_from_file(chunk_idx)?;

        // Insert into cache
        let evicted = {
            let mut cache = self.cache.write().map_err(|e| {
                CoreError::MemoryError(
                    ErrorContext::new(format!("Failed to lock cache for insert: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            cache.insert(chunk_idx, chunk_data.clone())
        };

        // Write back evicted dirty chunks
        if self.config.enable_write_back {
            for (evict_idx, evict_data, dirty) in evicted {
                if dirty {
                    self.write_chunk_to_file(evict_idx, &evict_data)?;
                }
            }
        }

        Ok(chunk_data)
    }

    /// Read a chunk from the file.
    fn read_chunk_from_file(&self, chunk_idx: usize) -> CoreResult<Vec<T>> {
        let element_size = std::mem::size_of::<T>();
        let chunk_start_element = chunk_idx * self.config.chunk_size;
        let chunk_end_element =
            (chunk_start_element + self.config.chunk_size).min(self.total_elements);
        let chunk_num_elements = chunk_end_element - chunk_start_element;

        let byte_offset = self.data_offset + chunk_start_element * element_size;
        let byte_count = chunk_num_elements * element_size;

        let mut file = self.file.lock().map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to lock file: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        file.seek(SeekFrom::Start(byte_offset as u64))
            .map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to seek to chunk {chunk_idx}: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        let mut raw_buf = vec![0u8; byte_count];
        file.read_exact(&mut raw_buf).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to read chunk {chunk_idx}: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        // Convert raw bytes to Vec<T>
        let mut result = vec![T::default(); chunk_num_elements];
        // SAFETY: We read exactly the right number of bytes and T is Copy
        unsafe {
            std::ptr::copy_nonoverlapping(
                raw_buf.as_ptr(),
                result.as_mut_ptr() as *mut u8,
                byte_count,
            );
        }

        Ok(result)
    }

    /// Write a chunk back to the file.
    fn write_chunk_to_file(&self, chunk_idx: usize, data: &[T]) -> CoreResult<()> {
        let element_size = std::mem::size_of::<T>();
        let chunk_start_element = chunk_idx * self.config.chunk_size;
        let byte_offset = self.data_offset + chunk_start_element * element_size;
        let raw_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        let mut file = self.file.lock().map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to lock file for write: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        file.seek(SeekFrom::Start(byte_offset as u64))
            .map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to seek for write: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

        file.write_all(raw_bytes).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to write chunk {chunk_idx}: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(())
    }

    /// Flush all dirty chunks to disk.
    pub fn flush(&self) -> CoreResult<()> {
        let dirty_indices: Vec<usize>;
        let dirty_data: Vec<(usize, Vec<T>)>;

        {
            let cache = self.cache.read().map_err(|e| {
                CoreError::MemoryError(
                    ErrorContext::new(format!("Failed to read-lock cache: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            dirty_indices = cache.dirty_chunks();
        }

        // Read out dirty chunk data
        {
            let mut cache = self.cache.write().map_err(|e| {
                CoreError::MemoryError(
                    ErrorContext::new(format!("Failed to write-lock cache: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
            dirty_data = dirty_indices
                .iter()
                .filter_map(|&idx| cache.get(idx).map(|data| (idx, data.to_vec())))
                .collect();

            // Mark all as clean
            for &idx in &dirty_indices {
                cache.mark_clean(idx);
            }
        }

        // Write to disk
        for (idx, data) in &dirty_data {
            self.write_chunk_to_file(*idx, data)?;
        }

        // Flush the file
        let mut file = self.file.lock().map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to lock file for flush: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        file.flush().map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to flush file: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(())
    }

    /// Create a streaming chunk iterator.
    ///
    /// This iterates through all chunks sequentially, loading each from disk
    /// (or cache) as needed.
    pub fn chunk_iter(&self) -> CoreResult<OocChunkIter<'_, T>> {
        Ok(OocChunkIter {
            array: self,
            current_chunk: 0,
        })
    }

    /// Process each chunk with a user-provided function.
    pub fn for_each_chunk<F>(&self, mut f: F) -> CoreResult<()>
    where
        F: FnMut(usize, &[T]) -> CoreResult<()>,
    {
        for chunk_idx in 0..self.num_chunks {
            let chunk_data = self.load_chunk(chunk_idx)?;
            f(chunk_idx, &chunk_data)?;
        }
        Ok(())
    }

    /// Apply a reduction over all chunks.
    pub fn reduce<R, F, C>(&self, init: R, mut combine: C, mut map_chunk: F) -> CoreResult<R>
    where
        F: FnMut(&[T]) -> R,
        C: FnMut(R, R) -> R,
    {
        let mut result = init;
        for chunk_idx in 0..self.num_chunks {
            let chunk_data = self.load_chunk(chunk_idx)?;
            let chunk_result = map_chunk(&chunk_data);
            result = combine(result, chunk_result);
        }
        Ok(result)
    }

    /// Create a new out-of-core array by applying a transformation to each element.
    pub fn map_to_file<U, F>(
        &self,
        output_path: &Path,
        config: AsyncOutOfCoreConfig,
        mut f: F,
    ) -> CoreResult<AsyncOutOfCoreArray<U>>
    where
        U: Copy + Default + Send + Sync + 'static,
        F: FnMut(T) -> U,
    {
        // Collect all transformed data
        let mut output_data = Vec::with_capacity(self.total_elements);
        for chunk_idx in 0..self.num_chunks {
            let chunk = self.load_chunk(chunk_idx)?;
            for &val in &chunk {
                output_data.push(f(val));
            }
        }

        AsyncOutOfCoreArray::from_slice(&output_data, output_path, config)
    }

    /// Clear the cache (flushing dirty chunks first).
    pub fn clear_cache(&self) -> CoreResult<()> {
        self.flush()?;
        let mut cache = self.cache.write().map_err(|e| {
            CoreError::MemoryError(
                ErrorContext::new(format!("Failed to write-lock cache: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;
        let _ = cache.clear();
        Ok(())
    }
}

impl<T: Copy + Default + Send + Sync + 'static> std::fmt::Debug for AsyncOutOfCoreArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncOutOfCoreArray")
            .field("path", &self.file_path)
            .field("total_elements", &self.total_elements)
            .field("num_chunks", &self.num_chunks)
            .field("chunk_size", &self.config.chunk_size)
            .field("element_size", &std::mem::size_of::<T>())
            .finish()
    }
}

// ============================================================================
// Chunk Iterator
// ============================================================================

/// Iterator that streams through chunks of an out-of-core array.
pub struct OocChunkIter<'a, T: Copy + Default + Send + Sync + 'static> {
    array: &'a AsyncOutOfCoreArray<T>,
    current_chunk: usize,
}

impl<'a, T: Copy + Default + Send + Sync + 'static> Iterator for OocChunkIter<'a, T> {
    type Item = CoreResult<Vec<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_chunk >= self.array.num_chunks {
            return None;
        }
        let chunk_idx = self.current_chunk;
        self.current_chunk += 1;
        Some(self.array.load_chunk(chunk_idx))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.num_chunks.saturating_sub(self.current_chunk);
        (remaining, Some(remaining))
    }
}

impl<'a, T: Copy + Default + Send + Sync + 'static> ExactSizeIterator for OocChunkIter<'a, T> {}

// ============================================================================
// Async support (tokio feature-gated)
// ============================================================================

#[cfg(feature = "async")]
mod async_support {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

    impl<T: Copy + Default + Send + Sync + 'static> AsyncOutOfCoreArray<T> {
        /// Asynchronously load a chunk from disk into the cache.
        ///
        /// This spawns a blocking I/O task on tokio's blocking thread pool.
        pub async fn load_chunk_async(&self, chunk_idx: usize) -> CoreResult<Vec<T>> {
            if chunk_idx >= self.num_chunks {
                return Err(CoreError::InvalidArgument(
                    ErrorContext::new(format!(
                        "Chunk index {} out of bounds (num_chunks: {})",
                        chunk_idx, self.num_chunks
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }

            // Check cache first
            {
                let mut cache = self.cache.write().map_err(|e| {
                    CoreError::MemoryError(
                        ErrorContext::new(format!("Failed to lock cache: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                if let Some(data) = cache.get(chunk_idx) {
                    return Ok(data.to_vec());
                }
            }

            // Read from file using tokio blocking task
            let file_clone = Arc::clone(&self.file);
            let config_chunk_size = self.config.chunk_size;
            let total_elements = self.total_elements;
            let data_offset = self.data_offset;
            let element_size = std::mem::size_of::<T>();

            let chunk_data = tokio::task::spawn_blocking(move || -> CoreResult<Vec<T>> {
                let chunk_start = chunk_idx * config_chunk_size;
                let chunk_end = (chunk_start + config_chunk_size).min(total_elements);
                let chunk_num_elements = chunk_end - chunk_start;
                let byte_offset = data_offset + chunk_start * element_size;
                let byte_count = chunk_num_elements * element_size;

                let mut file = file_clone.lock().map_err(|e| {
                    CoreError::IoError(
                        ErrorContext::new(format!("Failed to lock file: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;

                file.seek(SeekFrom::Start(byte_offset as u64))
                    .map_err(|e| {
                        CoreError::IoError(
                            ErrorContext::new(format!("Seek failed: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;

                let mut raw_buf = vec![0u8; byte_count];
                file.read_exact(&mut raw_buf).map_err(|e| {
                    CoreError::IoError(
                        ErrorContext::new(format!("Read failed: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;

                let mut result = vec![T::default(); chunk_num_elements];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        raw_buf.as_ptr(),
                        result.as_mut_ptr() as *mut u8,
                        byte_count,
                    );
                }
                Ok(result)
            })
            .await
            .map_err(|e| {
                CoreError::ComputationError(
                    ErrorContext::new(format!("Async task failed: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })??;

            // Insert into cache
            let evicted = {
                let mut cache = self.cache.write().map_err(|e| {
                    CoreError::MemoryError(
                        ErrorContext::new(format!("Failed to lock cache for insert: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                cache.insert(chunk_idx, chunk_data.clone())
            };

            // Write back evicted dirty chunks
            if self.config.enable_write_back {
                for (evict_idx, evict_data, dirty) in evicted {
                    if dirty {
                        self.write_chunk_to_file(evict_idx, &evict_data)?;
                    }
                }
            }

            Ok(chunk_data)
        }

        /// Asynchronously get a single element.
        pub async fn get_async(&self, index: usize) -> CoreResult<T> {
            if index >= self.total_elements {
                return Err(CoreError::InvalidArgument(
                    ErrorContext::new(format!(
                        "Index {} out of bounds for array of length {}",
                        index, self.total_elements
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }

            let chunk_idx = index / self.config.chunk_size;
            let offset_in_chunk = index % self.config.chunk_size;

            let chunk = self.load_chunk_async(chunk_idx).await?;

            if offset_in_chunk >= chunk.len() {
                return Err(CoreError::InvalidArgument(
                    ErrorContext::new(format!(
                        "Offset {} out of bounds within chunk",
                        offset_in_chunk
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }

            Ok(chunk[offset_in_chunk])
        }

        /// Asynchronously flush all dirty chunks.
        pub async fn flush_async(&self) -> CoreResult<()> {
            // Collect dirty data under lock
            let dirty_data: Vec<(usize, Vec<T>)>;
            {
                let mut cache = self.cache.write().map_err(|e| {
                    CoreError::MemoryError(
                        ErrorContext::new(format!("Failed to lock cache: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                let dirty_indices = cache.dirty_chunks();
                dirty_data = dirty_indices
                    .iter()
                    .filter_map(|&idx| cache.get(idx).map(|data| (idx, data.to_vec())))
                    .collect();
                for &idx in &dirty_indices {
                    cache.mark_clean(idx);
                }
            }

            // Write dirty chunks via blocking tasks
            let file_clone = Arc::clone(&self.file);
            let data_offset = self.data_offset;
            let config_chunk_size = self.config.chunk_size;
            let element_size = std::mem::size_of::<T>();

            for (idx, data) in dirty_data {
                let file_ref = Arc::clone(&file_clone);
                tokio::task::spawn_blocking(move || -> CoreResult<()> {
                    let chunk_start = idx * config_chunk_size;
                    let byte_offset = data_offset + chunk_start * element_size;
                    let byte_count = data.len() * element_size;

                    let raw_bytes = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_count)
                    };

                    let mut file = file_ref.lock().map_err(|e| {
                        CoreError::IoError(
                            ErrorContext::new(format!("Failed to lock file: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;

                    file.seek(SeekFrom::Start(byte_offset as u64))
                        .map_err(|e| {
                            CoreError::IoError(
                                ErrorContext::new(format!("Seek failed: {e}"))
                                    .with_location(ErrorLocation::new(file!(), line!())),
                            )
                        })?;

                    file.write_all(raw_bytes).map_err(|e| {
                        CoreError::IoError(
                            ErrorContext::new(format!("Write failed: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;

                    Ok(())
                })
                .await
                .map_err(|e| {
                    CoreError::ComputationError(
                        ErrorContext::new(format!("Async flush task failed: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })??;
            }

            // Final flush
            let file_ref = Arc::clone(&self.file);
            tokio::task::spawn_blocking(move || -> CoreResult<()> {
                let mut file = file_ref.lock().map_err(|e| {
                    CoreError::IoError(
                        ErrorContext::new(format!("Failed to lock file: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                file.flush().map_err(|e| {
                    CoreError::IoError(
                        ErrorContext::new(format!("Flush failed: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                Ok(())
            })
            .await
            .map_err(|e| {
                CoreError::ComputationError(
                    ErrorContext::new(format!("Async flush failed: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })??;

            Ok(())
        }

        /// Asynchronously prefetch chunks for upcoming access.
        pub async fn prefetch_async(&self, chunk_indices: &[usize]) -> CoreResult<()> {
            for &idx in chunk_indices {
                if idx < self.num_chunks {
                    self.load_chunk_async(idx).await?;
                }
            }
            Ok(())
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_read() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_basic.dat");

        let data: Vec<f64> = (0..10000).map(|i| i as f64 * 0.1).collect();
        let config = AsyncOutOfCoreConfig {
            chunk_size: 1000,
            max_cached_chunks: 4,
            max_cache_bytes: 64 * 1024,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config)
            .expect("Failed to create OOC array");

        assert_eq!(arr.len(), 10000);
        assert_eq!(arr.num_chunks(), 10);

        // Test random access
        let val = arr.get(5000).expect("Failed to get element");
        assert!((val - 500.0).abs() < 1e-10);

        let val2 = arr.get(9999).expect("Failed to get last element");
        assert!((val2 - 999.9).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_set_and_flush() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_write.dat");

        let data: Vec<f64> = vec![0.0; 5000];
        let config = AsyncOutOfCoreConfig {
            chunk_size: 1000,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config.clone())
            .expect("Failed to create");

        // Modify some elements
        arr.set(42, 123.456).expect("Failed to set");
        arr.set(3500, 789.012).expect("Failed to set");

        // Flush to disk
        arr.flush().expect("Failed to flush");
        drop(arr);

        // Reopen and verify
        let arr2 = AsyncOutOfCoreArray::<f64>::open(&path, config).expect("Failed to reopen");

        let val1 = arr2.get(42).expect("Failed to get");
        assert!((val1 - 123.456).abs() < 1e-10);

        let val2 = arr2.get(3500).expect("Failed to get");
        assert!((val2 - 789.012).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_chunk_iterator() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_iter.dat");

        let data: Vec<f64> = (0..2500).map(|i| i as f64).collect();
        let config = AsyncOutOfCoreConfig {
            chunk_size: 1000,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config).expect("Failed to create");

        let chunks: Vec<Vec<f64>> = arr
            .chunk_iter()
            .expect("Failed to create iter")
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to iterate");

        assert_eq!(chunks.len(), 3); // 2500 / 1000 = 2 full + 1 partial
        assert_eq!(chunks[0].len(), 1000);
        assert_eq!(chunks[1].len(), 1000);
        assert_eq!(chunks[2].len(), 500);

        // Verify data
        assert!((chunks[0][0] - 0.0).abs() < 1e-10);
        assert!((chunks[1][0] - 1000.0).abs() < 1e-10);
        assert!((chunks[2][0] - 2000.0).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_lru.dat");

        let data: Vec<f64> = (0..50000).map(|i| i as f64).collect();
        let config = AsyncOutOfCoreConfig {
            chunk_size: 1000,
            max_cached_chunks: 3,                // Only cache 3 chunks
            max_cache_bytes: 1024 * 1024 * 1024, // large byte limit
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config).expect("Failed to create");

        // Access chunks 0, 1, 2 (fills cache)
        let _ = arr.get(0).expect("Failed");
        let _ = arr.get(1000).expect("Failed");
        let _ = arr.get(2000).expect("Failed");

        // Access chunk 3 should evict chunk 0 (LRU)
        let _ = arr.get(3000).expect("Failed");

        let stats = arr.cache_statistics().expect("Failed to get stats");
        assert_eq!(stats.cached_chunks, 3); // Still only 3

        // Access chunk 0 again should trigger a miss (was evicted)
        let _ = arr.get(0).expect("Failed");
        let stats2 = arr.cache_statistics().expect("Failed to get stats");
        assert!(stats2.misses > 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_for_each_chunk() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_foreach.dat");

        let data: Vec<f64> = (0..3000).map(|i| i as f64).collect();
        let config = AsyncOutOfCoreConfig {
            chunk_size: 1000,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config).expect("Failed to create");

        let mut total_elements = 0usize;
        arr.for_each_chunk(|_chunk_idx, chunk_data| {
            total_elements += chunk_data.len();
            Ok(())
        })
        .expect("Failed to iterate");

        assert_eq!(total_elements, 3000);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_reduce() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_reduce.dat");

        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let config = AsyncOutOfCoreConfig {
            chunk_size: 30,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config).expect("Failed to create");

        let sum = arr
            .reduce(0.0, |a, b| a + b, |chunk| chunk.iter().sum::<f64>())
            .expect("Failed to reduce");

        let expected = 5050.0; // 1 + 2 + ... + 100
        assert!((sum - expected).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_map_to_file() {
        let dir = std::env::temp_dir();
        let input_path = dir.join("scirs2_ooc_test_map_in.dat");
        let output_path = dir.join("scirs2_ooc_test_map_out.dat");

        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let config = AsyncOutOfCoreConfig {
            chunk_size: 300,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &input_path, config.clone())
            .expect("Failed to create");

        let doubled = arr
            .map_to_file(&output_path, config, |x| x * 2.0)
            .expect("Failed to map");

        let val = doubled.get(500).expect("Failed to get");
        assert!((val - 1000.0).abs() < 1e-10);

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }

    #[test]
    fn test_out_of_bounds() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_oob.dat");

        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let config = AsyncOutOfCoreConfig {
            chunk_size: 10,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config).expect("Failed to create");

        assert!(arr.get(3).is_err());
        assert!(arr.get(100).is_err());
        assert!(arr.set(3, 42.0).is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_empty_array() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_empty.dat");

        let data: Vec<f64> = vec![];
        let config = AsyncOutOfCoreConfig {
            chunk_size: 100,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config).expect("Failed to create");

        assert_eq!(arr.len(), 0);
        assert!(arr.is_empty());
        assert_eq!(arr.num_chunks(), 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_cache_statistics() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_stats.dat");

        let data: Vec<f64> = (0..5000).map(|i| i as f64).collect();
        let config = AsyncOutOfCoreConfig {
            chunk_size: 1000,
            max_cached_chunks: 5,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config).expect("Failed to create");

        // First access - all misses
        let _ = arr.get(0).expect("Failed");
        let _ = arr.get(1500).expect("Failed");

        // Second access to same chunks - hits
        let _ = arr.get(100).expect("Failed"); // Same chunk as index 0
        let _ = arr.get(1200).expect("Failed"); // Same chunk as index 1500

        let stats = arr.cache_statistics().expect("Failed to get stats");
        assert_eq!(stats.cached_chunks, 2);
        assert!(stats.hits > 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_clear_cache() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_clear.dat");

        let data: Vec<f64> = (0..5000).map(|i| i as f64).collect();
        let config = AsyncOutOfCoreConfig {
            chunk_size: 1000,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config).expect("Failed to create");

        // Load some chunks
        let _ = arr.get(0).expect("Failed");
        let _ = arr.get(2000).expect("Failed");

        // Clear cache
        arr.clear_cache().expect("Failed to clear");

        let stats = arr.cache_statistics().expect("Failed to get stats");
        assert_eq!(stats.cached_chunks, 0);

        // Data should still be accessible
        let val = arr.get(2000).expect("Failed to get after clear");
        assert!((val - 2000.0).abs() < 1e-10);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_i32_type() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_i32.dat");

        let data: Vec<i32> = (0..5000).collect();
        let config = AsyncOutOfCoreConfig {
            chunk_size: 500,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config).expect("Failed to create");

        assert_eq!(arr.get(2500).expect("Failed"), 2500);
        assert_eq!(arr.get(4999).expect("Failed"), 4999);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_sequential_read_config() {
        let config = AsyncOutOfCoreConfig::sequential_read();
        assert!(config.prefetch_ahead > 0);
        assert!(!config.enable_write_back);
    }

    #[test]
    fn test_random_access_config() {
        let config = AsyncOutOfCoreConfig::random_access();
        assert_eq!(config.prefetch_ahead, 0);
        assert!(config.max_cached_chunks > 64);
    }

    #[test]
    fn test_large_dataset_config() {
        let config = AsyncOutOfCoreConfig::large_dataset();
        assert!(config.chunk_size > 100_000);
        assert!(config.max_cache_bytes >= 1024 * 1024 * 1024);
    }

    #[test]
    fn test_debug_format() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_debug.dat");

        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let config = AsyncOutOfCoreConfig {
            chunk_size: 10,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config).expect("Failed to create");

        let debug_str = format!("{:?}", arr);
        assert!(debug_str.contains("AsyncOutOfCoreArray"));
        assert!(debug_str.contains("total_elements"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_back_on_eviction() {
        let dir = std::env::temp_dir();
        let path = dir.join("scirs2_ooc_test_writeback.dat");

        let data: Vec<f64> = vec![0.0; 5000];
        let config = AsyncOutOfCoreConfig {
            chunk_size: 1000,
            max_cached_chunks: 2, // Only 2 cached
            max_cache_bytes: 1024 * 1024 * 1024,
            enable_write_back: true,
            ..Default::default()
        };

        let arr = AsyncOutOfCoreArray::from_slice(&data, &path, config.clone())
            .expect("Failed to create");

        // Modify chunk 0
        arr.set(42, 999.0).expect("Failed to set");

        // Load chunks 1 and 2 to evict chunk 0 (which is dirty)
        let _ = arr.get(1000).expect("Failed");
        let _ = arr.get(2000).expect("Failed");

        // Chunk 0 should have been written back when evicted
        drop(arr);

        // Reopen and verify
        let arr2 = AsyncOutOfCoreArray::<f64>::open(&path, config).expect("Failed to reopen");
        let val = arr2.get(42).expect("Failed to get");
        assert!(
            (val - 999.0).abs() < 1e-10,
            "Write-back should have persisted the modification"
        );

        let _ = std::fs::remove_file(&path);
    }
}
