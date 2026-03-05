//! WebWorker Communication Patterns for WASM
//!
//! This module provides **pure-Rust** types and serialization helpers to
//! coordinate work between the main JavaScript thread and one or more
//! `Worker` threads.  The actual `Worker` objects are managed on the JS
//! side; this module is responsible for:
//!
//! - Defining the message vocabulary (`WorkerMessage`)
//! - Serializing ndarray arrays into a flat, transferable format
//! - Deserializing results coming back from workers
//! - Modelling a `SharedBuffer` backed by a `SharedArrayBuffer`
//! - Providing a `WorkerPool` configuration / state-tracking struct
//!
//! ## Design notes
//!
//! All types cross the WASM/JS boundary as **JSON strings** so that they
//! work both with `postMessage` serialisation and with `SharedArrayBuffer`.
//! Where zero-copy is desired the caller should use `SharedBuffer` directly
//! and transfer the underlying `ArrayBuffer` handle.

use crate::error::WasmError;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Transferable array representation
// ---------------------------------------------------------------------------

/// A flat, transferable representation of an ndarray-compatible tensor.
///
/// The shape is stored separately so the recipient can reconstruct the
/// N-dimensional view after receiving the flat buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen(getter_with_clone)]
pub struct TransferableArray {
    /// Flat row-major data buffer.
    pub data: Vec<f64>,
    /// Shape of the array, e.g. `[3, 4]` for a 3×4 matrix.
    pub shape: Vec<usize>,
    /// Data type tag (always `"f64"` for this crate's arrays).
    pub dtype: String,
}

#[wasm_bindgen]
impl TransferableArray {
    /// Construct a `TransferableArray` from flat data and shape.
    ///
    /// # Errors
    ///
    /// Returns a JS error if `data.len()` does not match the product of
    /// `shape` elements.
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<TransferableArray, JsValue> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(WasmError::InvalidParameter(format!(
                "TransferableArray: data length {} does not match shape product {}",
                data.len(),
                expected,
            ))
            .into());
        }
        Ok(TransferableArray {
            data,
            shape,
            dtype: "f64".to_string(),
        })
    }

    /// Return the total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Return the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Serialise to a compact JSON string suitable for `postMessage`.
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Deserialise from a JSON string produced by `to_json`.
    ///
    /// # Errors
    ///
    /// Returns a JS error if the JSON is malformed or missing required fields.
    pub fn from_json(json: &str) -> Result<TransferableArray, JsValue> {
        serde_json::from_str(json)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }
}

// ---------------------------------------------------------------------------
// WorkerMessage – the message vocabulary
// ---------------------------------------------------------------------------

/// Kind of matrix operation requested by the main thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatrixOpKind {
    /// A × B
    Multiply,
    /// Transpose of A
    Transpose,
    /// Inverse of A (square matrices only)
    Inverse,
    /// Eigenvalue decomposition
    Eigenvalues,
    /// Singular value decomposition
    Svd,
}

/// Kind of statistics operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatsOpKind {
    /// Descriptive statistics (mean, std, …)
    Descriptive,
    /// Kolmogorov–Smirnov test
    KsTest,
    /// Independent t-test
    TTest,
    /// Pearson / Spearman correlation
    Correlation,
    /// Histogram binning.
    Histogram {
        /// Number of histogram bins to use.
        bins: usize,
    },
}

/// Kind of FFT operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FftOpKind {
    /// Forward FFT
    Forward,
    /// Inverse FFT
    Inverse,
    /// Short-time Fourier transform.
    Stft {
        /// Number of samples per STFT window.
        window_size: usize,
        /// Number of samples to advance between successive windows.
        hop_size: usize,
    },
    /// Power spectral density
    Psd,
}

/// A message sent **to** a Worker.
///
/// Each variant bundles the operation type with the payload arrays and any
/// additional parameters encoded as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op_type", content = "payload", rename_all = "snake_case")]
pub enum WorkerMessage {
    /// Matrix operation request.
    MatrixOp {
        /// Unique task identifier (echoed back in the result).
        task_id: String,
        /// Operation to perform.
        op: MatrixOpKind,
        /// Primary input array.
        a: TransferableArray,
        /// Secondary input array (optional, used for Multiply).
        b: Option<TransferableArray>,
    },

    /// Statistics operation request.
    StatsOp {
        /// Unique task identifier.
        task_id: String,
        /// Operation to perform.
        op: StatsOpKind,
        /// Data array(s) – first sample / primary data.
        data: TransferableArray,
        /// Optional second sample (for two-sample tests).
        data2: Option<TransferableArray>,
    },

    /// FFT operation request.
    FftOp {
        /// Unique task identifier.
        task_id: String,
        /// Operation to perform.
        op: FftOpKind,
        /// Input signal (real-valued or interleaved complex).
        signal: TransferableArray,
        /// Whether signal data is interleaved complex `[re, im, re, im, …]`.
        is_complex: bool,
    },

    /// Cancel a pending task by ID.
    Cancel {
        /// Task ID to cancel.
        task_id: String,
    },

    /// Worker → main thread: computation result.
    Result {
        /// Task ID from the original request.
        task_id: String,
        /// Whether the computation succeeded.
        success: bool,
        /// Output array (present on success).
        data: Option<TransferableArray>,
        /// Additional metadata / statistics (JSON string).
        metadata: Option<String>,
        /// Error message (present on failure).
        error: Option<String>,
    },

    /// Worker → main thread: progress notification.
    Progress {
        /// Task ID.
        task_id: String,
        /// Progress percentage 0–100.
        percent: f64,
        /// Human-readable status.
        status: String,
    },
}

impl WorkerMessage {
    /// Serialise a `WorkerMessage` to a JSON string.
    ///
    /// The result can be passed directly to `Worker.postMessage()`.
    ///
    /// Note: This is not exposed via `#[wasm_bindgen]` because `WorkerMessage`
    /// is a complex enum with data variants, which wasm-bindgen does not
    /// support as a receiver type.  JS callers should use the free-standing
    /// [`serialize_worker_message`] helper instead.
    pub fn serialize_message(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }
}

/// Serialize a [`WorkerMessage`] (given as a JSON string) back to a validated
/// JSON string.
///
/// This is the JS-friendly counterpart of [`WorkerMessage::serialize_message`].
/// Pass the output directly to `Worker.postMessage()`.
///
/// # Errors
///
/// Returns a JS error if the input is not valid `WorkerMessage` JSON.
#[wasm_bindgen]
pub fn serialize_worker_message(json: &str) -> Result<String, JsValue> {
    let msg: WorkerMessage = serde_json::from_str(json)
        .map_err(|e| WasmError::SerializationError(e.to_string()))?;
    serde_json::to_string(&msg)
        .map_err(|e| WasmError::SerializationError(e.to_string()).into())
}

// ---------------------------------------------------------------------------
// Free-standing serialization / deserialization helpers
// ---------------------------------------------------------------------------

/// Serialize an ndarray-compatible flat f64 array to a `TransferableArray`.
///
/// # Arguments
///
/// * `data`  – flat row-major data
/// * `shape` – dimension sizes (product must equal `data.len()`)
///
/// # Errors
///
/// Returns a JS error if the shape is inconsistent with the data length.
#[wasm_bindgen]
pub fn serialize_for_worker(data: &[f64], shape: Vec<usize>) -> Result<String, JsValue> {
    let arr = TransferableArray::new(data.to_vec(), shape)?;
    arr.to_json()
}

/// Deserialize the JSON string returned by a Worker into a `TransferableArray`.
///
/// Validates that the shape product matches the data buffer length.
///
/// # Errors
///
/// Returns a JS error if the JSON is malformed or the shape is inconsistent.
#[wasm_bindgen]
pub fn deserialize_from_worker(json: &str) -> Result<TransferableArray, JsValue> {
    let arr = TransferableArray::from_json(json)?;
    // Re-validate shape consistency after round-trip.
    let expected: usize = arr.shape.iter().product();
    if expected != arr.data.len() {
        return Err(WasmError::InvalidParameter(format!(
            "deserialize_from_worker: shape product {} ≠ data length {}",
            expected,
            arr.data.len(),
        ))
        .into());
    }
    Ok(arr)
}

/// Parse a `WorkerMessage` from a JSON string (e.g. received via `onmessage`).
///
/// # Errors
///
/// Returns a JS error if the JSON is malformed or the message type is unknown.
#[wasm_bindgen]
pub fn parse_worker_message(json: &str) -> Result<JsValue, JsValue> {
    let msg: WorkerMessage = serde_json::from_str(json)
        .map_err(|e| WasmError::SerializationError(e.to_string()))?;

    serde_wasm_bindgen::to_value(&msg)
        .map_err(|e| WasmError::SerializationError(e.to_string()).into())
}

// ---------------------------------------------------------------------------
// SharedBuffer – SharedArrayBuffer-backed data exchange
// ---------------------------------------------------------------------------

/// Configuration and metadata for a `SharedArrayBuffer`-backed buffer.
///
/// The actual `SharedArrayBuffer` is held on the JavaScript side; this struct
/// carries the metadata needed to interpret its contents from Rust.
///
/// ## Layout
///
/// ```text
/// [ header (32 bytes) | data (element_count × element_size bytes) ]
///
/// Header layout (all u32 little-endian):
///   offset 0 : magic number 0x53_43_52_53 ("SCRS")
///   offset 4 : state (0 = idle, 1 = writing, 2 = readable)
///   offset 8 : element_count (u32)
///   offset 12: element_size  (u32) – bytes per element (4 for f32, 8 for f64)
///   offset 16: shape_len     (u32) – number of dimensions
///   offsets 20–31: shape[0..3] (u32 × 3, padded to 4)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen(getter_with_clone)]
pub struct SharedBuffer {
    /// Total byte length of the `SharedArrayBuffer` (header + data).
    pub byte_length: usize,
    /// Number of data elements.
    pub element_count: usize,
    /// Bytes per element (4 = f32, 8 = f64).
    pub element_size: usize,
    /// Shape of the logical tensor.
    pub shape: Vec<usize>,
    /// Human-readable label for debugging.
    pub label: String,
}

#[wasm_bindgen]
impl SharedBuffer {
    /// Create a new `SharedBuffer` descriptor for a tensor with the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape`        – dimension sizes
    /// * `element_size` – bytes per element: 4 (f32) or 8 (f64)
    /// * `label`        – optional debug label
    ///
    /// # Errors
    ///
    /// Returns a JS error if `element_size` is not 4 or 8, or if the shape is empty.
    #[wasm_bindgen(constructor)]
    pub fn new(
        shape: Vec<usize>,
        element_size: usize,
        label: String,
    ) -> Result<SharedBuffer, JsValue> {
        if shape.is_empty() {
            return Err(
                WasmError::InvalidParameter("SharedBuffer: shape must not be empty".to_string())
                    .into(),
            );
        }
        if element_size != 4 && element_size != 8 {
            return Err(WasmError::InvalidParameter(format!(
                "SharedBuffer: element_size must be 4 (f32) or 8 (f64), got {}",
                element_size
            ))
            .into());
        }

        let element_count: usize = shape.iter().product();
        // 32-byte header + data region.
        let byte_length = 32 + element_count * element_size;

        Ok(SharedBuffer {
            byte_length,
            element_count,
            element_size,
            shape,
            label,
        })
    }

    /// Return the byte offset at which data begins (always 32).
    pub fn data_offset(&self) -> usize {
        32
    }

    /// Serialise the descriptor to JSON for transfer to a Worker.
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Reconstruct a `SharedBuffer` descriptor from a JSON string.
    pub fn from_json(json: &str) -> Result<SharedBuffer, JsValue> {
        serde_json::from_str(json)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Generate the Rust representation of the header bytes as a `Vec<u8>`.
    ///
    /// The caller should write these bytes to offset 0 of the
    /// `SharedArrayBuffer` before signalling workers.
    pub fn build_header(&self) -> Vec<u8> {
        let mut header = vec![0u8; 32];

        // Magic: "SCRS"
        header[0] = 0x53;
        header[1] = 0x43;
        header[2] = 0x52;
        header[3] = 0x53;

        // State: idle = 0
        let state: u32 = 0;
        header[4..8].copy_from_slice(&state.to_le_bytes());

        // element_count
        let count = self.element_count as u32;
        header[8..12].copy_from_slice(&count.to_le_bytes());

        // element_size
        let esize = self.element_size as u32;
        header[12..16].copy_from_slice(&esize.to_le_bytes());

        // shape_len
        let shape_len = self.shape.len().min(3) as u32;
        header[16..20].copy_from_slice(&shape_len.to_le_bytes());

        // shape[0..3] padded
        for (i, &dim) in self.shape.iter().take(3).enumerate() {
            let d = dim as u32;
            let off = 20 + i * 4;
            header[off..off + 4].copy_from_slice(&d.to_le_bytes());
        }

        header
    }
}

// ---------------------------------------------------------------------------
// WorkerPool – pool configuration / tracking abstraction
// ---------------------------------------------------------------------------

/// Lifecycle state of a single Worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerState {
    /// Worker is idle and ready to accept tasks.
    Idle,
    /// Worker is processing a task.
    Busy,
    /// Worker has terminated.
    Terminated,
    /// Worker failed to start or encountered a fatal error.
    Error,
}

/// Metadata tracked per Worker by the pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerEntry {
    /// Unique worker index within the pool (0-based).
    pub index: usize,
    /// Current lifecycle state.
    pub state: WorkerState,
    /// ID of the task currently being processed (if any).
    pub current_task_id: Option<String>,
    /// Total tasks completed since pool creation.
    pub tasks_completed: u64,
    /// Total tasks failed.
    pub tasks_failed: u64,
}

/// Pool-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen(getter_with_clone)]
pub struct WorkerPoolConfig {
    /// Desired number of workers.
    pub worker_count: usize,
    /// URL of the worker script (relative or absolute).
    pub worker_script_url: String,
    /// Maximum number of tasks queued before backpressure is applied.
    pub max_queue_depth: usize,
    /// Timeout in milliseconds before a stalled task is considered failed.
    pub task_timeout_ms: u64,
}

#[wasm_bindgen]
impl WorkerPoolConfig {
    /// Create a new pool configuration.
    ///
    /// # Arguments
    ///
    /// * `worker_count`      – number of workers to spawn (clamped to 1..=64)
    /// * `worker_script_url` – URL of the worker entry-point script
    ///
    /// # Errors
    ///
    /// Returns a JS error if `worker_script_url` is empty.
    #[wasm_bindgen(constructor)]
    pub fn new(worker_count: usize, worker_script_url: String) -> Result<WorkerPoolConfig, JsValue> {
        if worker_script_url.is_empty() {
            return Err(WasmError::InvalidParameter(
                "WorkerPoolConfig: worker_script_url must not be empty".to_string(),
            )
            .into());
        }
        let worker_count = worker_count.clamp(1, 64);
        Ok(WorkerPoolConfig {
            worker_count,
            worker_script_url,
            max_queue_depth: 256,
            task_timeout_ms: 30_000,
        })
    }

    /// Override the maximum task queue depth.
    pub fn with_max_queue_depth(mut self, depth: usize) -> WorkerPoolConfig {
        self.max_queue_depth = depth.max(1);
        self
    }

    /// Override the task timeout.
    pub fn with_task_timeout_ms(mut self, ms: u64) -> WorkerPoolConfig {
        self.task_timeout_ms = ms;
        self
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }
}

/// A pure-Rust abstraction that tracks the state of a pool of Web Workers.
///
/// This struct does **not** own JS `Worker` objects; it only stores
/// bookkeeping metadata.  The actual worker lifecycle is driven from
/// JavaScript using the configuration emitted by `to_init_script()`.
#[wasm_bindgen]
pub struct WorkerPool {
    config: WorkerPoolConfig,
    workers: Vec<WorkerEntry>,
    pending_queue: Vec<String>, // pending task IDs
    next_task_seq: u64,
}

#[wasm_bindgen]
impl WorkerPool {
    /// Create a new pool tracker from a `WorkerPoolConfig`.
    #[wasm_bindgen(constructor)]
    pub fn new(config: WorkerPoolConfig) -> WorkerPool {
        let workers: Vec<WorkerEntry> = (0..config.worker_count)
            .map(|i| WorkerEntry {
                index: i,
                state: WorkerState::Idle,
                current_task_id: None,
                tasks_completed: 0,
                tasks_failed: 0,
            })
            .collect();

        WorkerPool {
            config,
            workers,
            pending_queue: Vec::new(),
            next_task_seq: 0,
        }
    }

    /// Generate a unique task ID and optionally enqueue it.
    ///
    /// Returns the generated ID as a `String`.
    pub fn enqueue_task(&mut self, op_type: &str) -> String {
        let task_id = format!("{}-{}", op_type, self.next_task_seq);
        self.next_task_seq += 1;
        self.pending_queue.push(task_id.clone());
        task_id
    }

    /// Try to assign a pending task to an idle worker.
    ///
    /// Returns the task ID assigned, or `None` if all workers are busy or
    /// the queue is empty.
    pub fn try_dispatch(&mut self) -> Option<String> {
        if self.pending_queue.is_empty() {
            return None;
        }

        let idle_idx = self.workers.iter().position(|w| w.state == WorkerState::Idle)?;
        let task_id = self.pending_queue.remove(0);

        self.workers[idle_idx].state = WorkerState::Busy;
        self.workers[idle_idx].current_task_id = Some(task_id.clone());

        Some(task_id)
    }

    /// Mark a task as completed (called when the Worker sends a Result message).
    ///
    /// # Returns
    ///
    /// `true` if the worker was found and updated, `false` if the task ID was unknown.
    pub fn task_completed(&mut self, task_id: &str, success: bool) -> bool {
        if let Some(worker) = self
            .workers
            .iter_mut()
            .find(|w| w.current_task_id.as_deref() == Some(task_id))
        {
            if success {
                worker.tasks_completed += 1;
            } else {
                worker.tasks_failed += 1;
            }
            worker.state = WorkerState::Idle;
            worker.current_task_id = None;
            return true;
        }
        false
    }

    /// Return pool statistics as a JSON string.
    pub fn stats(&self) -> Result<String, JsValue> {
        let idle = self.workers.iter().filter(|w| w.state == WorkerState::Idle).count();
        let busy = self.workers.iter().filter(|w| w.state == WorkerState::Busy).count();
        let completed: u64 = self.workers.iter().map(|w| w.tasks_completed).sum();
        let failed: u64 = self.workers.iter().map(|w| w.tasks_failed).sum();

        let stats = serde_json::json!({
            "worker_count": self.config.worker_count,
            "idle": idle,
            "busy": busy,
            "pending_queue_depth": self.pending_queue.len(),
            "total_completed": completed,
            "total_failed": failed,
        });

        serde_json::to_string(&stats)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Return the number of workers in the pool.
    pub fn worker_count(&self) -> usize {
        self.config.worker_count
    }

    /// Return the current depth of the pending task queue.
    pub fn pending_count(&self) -> usize {
        self.pending_queue.len()
    }

    /// Generate a minimal JavaScript initialisation snippet that creates the
    /// workers using the stored configuration.
    ///
    /// The caller should `eval()` or include this snippet in their worker
    /// harness.  The snippet exposes a `pool` global array of `Worker` objects.
    pub fn to_init_script(&self) -> Result<String, JsValue> {
        let script = format!(
            r#"
// Auto-generated by WorkerPool::to_init_script
const POOL_CONFIG = {{
  workerCount: {worker_count},
  workerScriptUrl: {url_json},
  maxQueueDepth: {max_queue},
  taskTimeoutMs: {timeout_ms},
}};
const pool = Array.from({{ length: POOL_CONFIG.workerCount }}, () =>
  new Worker(POOL_CONFIG.workerScriptUrl, {{ type: 'module' }})
);
"#,
            worker_count = self.config.worker_count,
            url_json = serde_json::to_string(&self.config.worker_script_url)
                .map_err(|e| WasmError::SerializationError(e.to_string()))?,
            max_queue = self.config.max_queue_depth,
            timeout_ms = self.config.task_timeout_ms,
        );
        Ok(script)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transferable_array_roundtrip() {
        let arr = TransferableArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .expect("construction ok");
        let json = arr.to_json().expect("to_json ok");
        let recovered = TransferableArray::from_json(&json).expect("from_json ok");
        assert_eq!(recovered.data, arr.data);
        assert_eq!(recovered.shape, vec![2, 2]);
    }

    #[test]
    fn test_transferable_array_shape_mismatch() {
        let result = TransferableArray::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialize_for_worker() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let json = serialize_for_worker(&data, vec![3]).expect("serialize ok");
        let recovered = deserialize_from_worker(&json).expect("deserialize ok");
        assert_eq!(recovered.data, data);
    }

    #[test]
    fn test_shared_buffer_header() {
        let buf = SharedBuffer::new(vec![4, 4], 4, "test".to_string())
            .expect("SharedBuffer ok");
        assert_eq!(buf.element_count, 16);
        assert_eq!(buf.element_size, 4);
        assert_eq!(buf.byte_length, 32 + 64);

        let header = buf.build_header();
        assert_eq!(&header[0..4], b"SCRS");
        // State should be 0 (idle)
        assert_eq!(&header[4..8], &0u32.to_le_bytes());
        // element_count = 16
        assert_eq!(&header[8..12], &16u32.to_le_bytes());
    }

    #[test]
    fn test_shared_buffer_bad_element_size() {
        let result = SharedBuffer::new(vec![4], 3, "bad".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_worker_pool_dispatch() {
        let config =
            WorkerPoolConfig::new(2, "/worker.js".to_string()).expect("config ok");
        let mut pool = WorkerPool::new(config);

        let task1 = pool.enqueue_task("matrix_op");
        let task2 = pool.enqueue_task("stats_op");

        assert_eq!(pool.pending_count(), 2);

        let dispatched = pool.try_dispatch().expect("dispatch ok");
        assert_eq!(dispatched, task1);
        assert_eq!(pool.pending_count(), 1);

        let dispatched2 = pool.try_dispatch().expect("dispatch ok");
        assert_eq!(dispatched2, task2);
        assert_eq!(pool.pending_count(), 0);

        // Both workers busy, nothing to dispatch.
        assert!(pool.try_dispatch().is_none());

        // Complete task1.
        assert!(pool.task_completed(&task1, true));

        // Now one worker is idle again.
        let task3 = pool.enqueue_task("fft_op");
        let d = pool.try_dispatch().expect("dispatch ok");
        assert_eq!(d, task3);
    }

    #[test]
    fn test_pool_init_script() {
        let config =
            WorkerPoolConfig::new(4, "https://example.com/worker.js".to_string())
                .expect("config ok");
        let pool = WorkerPool::new(config);
        let script = pool.to_init_script().expect("script ok");
        assert!(script.contains("workerCount: 4"));
        assert!(script.contains("https://example.com/worker.js"));
    }

    #[test]
    fn test_worker_pool_config_clamp() {
        // 0 workers should clamp to 1
        let config = WorkerPoolConfig::new(0, "/w.js".to_string()).expect("ok");
        assert_eq!(config.worker_count, 1);

        // 1000 workers should clamp to 64
        let config2 = WorkerPoolConfig::new(1000, "/w.js".to_string()).expect("ok");
        assert_eq!(config2.worker_count, 64);
    }

    #[test]
    fn test_worker_message_serialization() {
        let arr = TransferableArray::new(vec![1.0, 2.0], vec![2]).expect("ok");
        let msg = WorkerMessage::MatrixOp {
            task_id: "task-0".to_string(),
            op: MatrixOpKind::Transpose,
            a: arr,
            b: None,
        };
        let json = msg.serialize_message().expect("serialize ok");
        assert!(json.contains("task-0"));
        assert!(json.contains("transpose"));
    }
}
