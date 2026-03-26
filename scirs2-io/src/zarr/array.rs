//! ZarrArray: the main interface for reading and writing chunked Zarr arrays.

use std::collections::HashMap;

use crate::error::{IoError, Result};

use super::codecs::{BytesCodec, CodecPipeline, Endian, ZstdCodec};
use super::metadata::{ArrayMetadataV2, ArrayMetadataV3, CodecMetadata, CompressorV2};
use super::store::{chunk_key_v2, chunk_key_v3, DirectoryStore};
use super::{DataType, ZarrVersion};

/// A chunked N-dimensional array stored in a Zarr directory store.
#[derive(Debug)]
pub struct ZarrArray {
    store: DirectoryStore,
    /// Path prefix within the store (e.g. `"myarray"`).
    path: String,
    /// Zarr version.
    version: ZarrVersion,
    /// Array shape.
    shape: Vec<u64>,
    /// Chunk shape.
    chunks: Vec<u64>,
    /// Scalar data type.
    dtype: DataType,
    /// Fill value as f64 (used when a chunk does not exist).
    fill_value: f64,
    /// Codec pipeline for encoding/decoding chunks.
    pipeline: CodecPipeline,
    /// Memory layout order (`"C"` or `"F"`).
    order: String,
    /// Dimension separator for chunk keys (v2).
    dim_separator: String,
    /// Chunk key separator for v3.
    v3_separator: String,
    /// Whether compression is enabled.
    compressed: bool,
}

impl ZarrArray {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Create a new Zarr array.
    pub fn create(
        store: DirectoryStore,
        path: &str,
        shape: Vec<u64>,
        chunks: Vec<u64>,
        dtype: DataType,
        version: ZarrVersion,
    ) -> Result<Self> {
        Self::create_with_options(store, path, shape, chunks, dtype, version, 0.0, true)
    }

    /// Create a new Zarr array with explicit fill value and compression option.
    pub fn create_with_options(
        store: DirectoryStore,
        path: &str,
        shape: Vec<u64>,
        chunks: Vec<u64>,
        dtype: DataType,
        version: ZarrVersion,
        fill_value: f64,
        compress: bool,
    ) -> Result<Self> {
        if shape.len() != chunks.len() {
            return Err(IoError::FormatError(format!(
                "Shape ndim ({}) != chunks ndim ({})",
                shape.len(),
                chunks.len()
            )));
        }

        let pipeline = build_pipeline(dtype, compress);

        let arr = Self {
            store,
            path: path.to_string(),
            version,
            shape: shape.clone(),
            chunks: chunks.clone(),
            dtype,
            fill_value,
            pipeline,
            order: "C".to_string(),
            dim_separator: ".".to_string(),
            v3_separator: "/".to_string(),
            compressed: compress,
        };

        // Write metadata
        match version {
            ZarrVersion::V2 => arr.write_v2_metadata()?,
            ZarrVersion::V3 => arr.write_v3_metadata()?,
        }

        Ok(arr)
    }

    /// Open an existing Zarr array from a store.
    pub fn open(store: DirectoryStore, path: &str) -> Result<Self> {
        // Try v3 first (zarr.json)
        let v3_key = if path.is_empty() {
            "zarr.json".to_string()
        } else {
            format!("{path}/zarr.json")
        };
        if store.exists(&v3_key) {
            let data = store.get(&v3_key)?;
            let meta = ArrayMetadataV3::from_json(&data)?;
            return Self::from_v3_metadata(store, path, &meta);
        }

        // Try v2 (.zarray)
        let v2_key = if path.is_empty() {
            ".zarray".to_string()
        } else {
            format!("{path}/.zarray")
        };
        if store.exists(&v2_key) {
            let data = store.get(&v2_key)?;
            let meta = ArrayMetadataV2::from_json(&data)?;
            return Self::from_v2_metadata(store, path, &meta);
        }

        Err(IoError::FileNotFound(format!(
            "No Zarr array metadata found at path '{path}'"
        )))
    }

    fn from_v2_metadata(store: DirectoryStore, path: &str, meta: &ArrayMetadataV2) -> Result<Self> {
        let dtype = meta.data_type()?;
        let compressed = meta.compressor.is_some();
        let pipeline = build_pipeline(dtype, compressed);
        let fill_value = meta.fill_value.as_f64().unwrap_or(0.0);

        Ok(Self {
            store,
            path: path.to_string(),
            version: ZarrVersion::V2,
            shape: meta.shape.clone(),
            chunks: meta.chunks.clone(),
            dtype,
            fill_value,
            pipeline,
            order: meta.order.clone(),
            dim_separator: meta.dimension_separator.clone(),
            v3_separator: "/".to_string(),
            compressed,
        })
    }

    fn from_v3_metadata(store: DirectoryStore, path: &str, meta: &ArrayMetadataV3) -> Result<Self> {
        let dtype = meta.data_type_parsed()?;
        let chunk_shape = meta.chunk_shape()?.to_vec();
        let compressed = meta
            .codecs
            .iter()
            .any(|c| c.name == "zstd" || c.name == "gzip");
        let pipeline = build_pipeline(dtype, compressed);
        let fill_value = meta
            .fill_value
            .as_ref()
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let separator = meta
            .chunk_key_encoding
            .as_ref()
            .and_then(|e| e.configuration.as_ref())
            .map(|c| c.separator.clone())
            .unwrap_or_else(|| "/".to_string());

        Ok(Self {
            store,
            path: path.to_string(),
            version: ZarrVersion::V3,
            shape: meta.shape.clone(),
            chunks: chunk_shape,
            dtype,
            fill_value,
            pipeline,
            order: "C".to_string(),
            dim_separator: ".".to_string(),
            v3_separator: separator,
            compressed,
        })
    }

    fn write_v2_metadata(&self) -> Result<()> {
        let compressor = if self.compressed {
            Some(CompressorV2 {
                id: "zstd".to_string(),
                level: Some(3),
                extra: HashMap::new(),
            })
        } else {
            None
        };

        let meta = ArrayMetadataV2::new(
            self.shape.clone(),
            self.chunks.clone(),
            self.dtype,
            compressor,
            serde_json::json!(self.fill_value),
        );
        let json = meta.to_json()?;
        let key = if self.path.is_empty() {
            ".zarray".to_string()
        } else {
            format!("{}/.zarray", self.path)
        };
        self.store.set(&key, &json)
    }

    fn write_v3_metadata(&self) -> Result<()> {
        let mut codecs = vec![CodecMetadata {
            name: "bytes".to_string(),
            configuration: Some(serde_json::json!({"endian": "little"})),
        }];
        if self.compressed {
            codecs.push(CodecMetadata {
                name: "zstd".to_string(),
                configuration: Some(serde_json::json!({"level": 3})),
            });
        }

        let meta = ArrayMetadataV3::new_array(
            self.shape.clone(),
            self.chunks.clone(),
            self.dtype,
            serde_json::json!(self.fill_value),
            codecs,
        );
        let json = meta.to_json()?;
        let key = if self.path.is_empty() {
            "zarr.json".to_string()
        } else {
            format!("{}/zarr.json", self.path)
        };
        self.store.set(&key, &json)
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// Array shape.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Chunk shape.
    pub fn chunk_shape(&self) -> &[u64] {
        &self.chunks
    }

    /// Data type.
    pub fn data_type(&self) -> DataType {
        self.dtype
    }

    /// Zarr version.
    pub fn version(&self) -> ZarrVersion {
        self.version
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Number of chunks along each dimension.
    pub fn num_chunks(&self) -> Vec<u64> {
        self.shape
            .iter()
            .zip(self.chunks.iter())
            .map(|(&s, &c)| (s + c - 1) / c)
            .collect()
    }

    // ── Chunk I/O ────────────────────────────────────────────────────────

    /// Compute the chunk key for given chunk coordinates.
    fn chunk_key(&self, coords: &[u64]) -> String {
        match self.version {
            ZarrVersion::V2 => chunk_key_v2(&self.path, coords, &self.dim_separator),
            ZarrVersion::V3 => chunk_key_v3(&self.path, coords, &self.v3_separator),
        }
    }

    /// Number of elements in a single chunk.
    fn chunk_num_elements(&self) -> usize {
        self.chunks.iter().map(|&c| c as usize).product()
    }

    /// Read a single chunk. Returns fill values if the chunk does not exist.
    pub fn read_chunk(&self, coords: &[u64]) -> Result<Vec<f64>> {
        let key = self.chunk_key(coords);
        let n = self.chunk_num_elements();

        if !self.store.exists(&key) {
            return Ok(vec![self.fill_value; n]);
        }

        let raw = self.store.get(&key)?;
        let decoded = self.pipeline.decode(&raw)?;
        bytes_to_f64(&decoded, self.dtype, n)
    }

    /// Write a single chunk.
    pub fn write_chunk(&self, coords: &[u64], data: &[f64]) -> Result<()> {
        let n = self.chunk_num_elements();
        if data.len() != n {
            return Err(IoError::FormatError(format!(
                "Chunk data length {} != expected {}",
                data.len(),
                n
            )));
        }
        let raw = f64_to_bytes(data, self.dtype)?;
        let encoded = self.pipeline.encode(&raw)?;
        let key = self.chunk_key(coords);
        self.store.set(&key, &encoded)
    }

    // ── Slice I/O (arbitrary selection across chunk boundaries) ──────────

    /// Read a rectangular selection from the array.
    ///
    /// `start` and `end` are per-dimension half-open ranges `[start, end)`.
    /// Returns data in C-order (row-major).
    pub fn get(&self, start: &[u64], end: &[u64]) -> Result<Vec<f64>> {
        let ndim = self.ndim();
        if start.len() != ndim || end.len() != ndim {
            return Err(IoError::FormatError(format!(
                "Selection dimensionality mismatch: ndim={ndim}, start.len={}, end.len={}",
                start.len(),
                end.len()
            )));
        }
        for d in 0..ndim {
            if start[d] >= end[d] || end[d] > self.shape[d] {
                return Err(IoError::FormatError(format!(
                    "Invalid selection range in dim {d}: [{}, {}) out of [0, {})",
                    start[d], end[d], self.shape[d]
                )));
            }
        }

        let sel_shape: Vec<u64> = (0..ndim).map(|d| end[d] - start[d]).collect();
        let total: usize = sel_shape.iter().map(|&s| s as usize).product();
        let mut result = vec![0.0f64; total];

        // Iterate over all chunks that overlap the selection
        let chunk_start: Vec<u64> = (0..ndim).map(|d| start[d] / self.chunks[d]).collect();
        let chunk_end: Vec<u64> = (0..ndim)
            .map(|d| (end[d] + self.chunks[d] - 1) / self.chunks[d])
            .collect();

        let mut chunk_coords = chunk_start.clone();
        loop {
            let chunk_data = self.read_chunk(&chunk_coords)?;

            // Compute overlap between this chunk and the selection
            let c_start: Vec<u64> = (0..ndim)
                .map(|d| chunk_coords[d] * self.chunks[d])
                .collect();

            let overlap_start: Vec<u64> = (0..ndim).map(|d| start[d].max(c_start[d])).collect();
            let overlap_end: Vec<u64> = (0..ndim)
                .map(|d| end[d].min(c_start[d] + self.chunks[d]))
                .collect();

            // Copy elements from chunk into result
            let overlap_shape: Vec<u64> = (0..ndim)
                .map(|d| overlap_end[d] - overlap_start[d])
                .collect();
            let overlap_total: usize = overlap_shape.iter().map(|&s| s as usize).product();

            for linear in 0..overlap_total {
                // Convert to multi-dim index within overlap
                let mut multi = vec![0u64; ndim];
                let mut rem = linear;
                for d in (0..ndim).rev() {
                    multi[d] = (rem % overlap_shape[d] as usize) as u64;
                    rem /= overlap_shape[d] as usize;
                }

                // Index into chunk data (C-order within chunk)
                let chunk_idx: Vec<u64> = (0..ndim)
                    .map(|d| overlap_start[d] + multi[d] - c_start[d])
                    .collect();
                let chunk_linear = c_order_index(&chunk_idx, &self.chunks);

                // Index into result (C-order within selection)
                let sel_idx: Vec<u64> = (0..ndim)
                    .map(|d| overlap_start[d] + multi[d] - start[d])
                    .collect();
                let sel_linear = c_order_index(&sel_idx, &sel_shape);

                result[sel_linear] = chunk_data[chunk_linear];
            }

            // Advance chunk coordinates (last dimension first)
            if !advance_coords(&mut chunk_coords, &chunk_start, &chunk_end, ndim) {
                break;
            }
        }

        Ok(result)
    }

    /// Write a rectangular selection into the array.
    ///
    /// `start` and `end` define the half-open range per dimension.
    /// `data` is in C-order.
    pub fn set(&self, start: &[u64], end: &[u64], data: &[f64]) -> Result<()> {
        let ndim = self.ndim();
        if start.len() != ndim || end.len() != ndim {
            return Err(IoError::FormatError(format!(
                "Selection dimensionality mismatch: ndim={ndim}, start.len={}, end.len={}",
                start.len(),
                end.len()
            )));
        }
        let sel_shape: Vec<u64> = (0..ndim).map(|d| end[d] - start[d]).collect();
        let total: usize = sel_shape.iter().map(|&s| s as usize).product();
        if data.len() != total {
            return Err(IoError::FormatError(format!(
                "Data length {} != selection size {}",
                data.len(),
                total
            )));
        }

        let chunk_start: Vec<u64> = (0..ndim).map(|d| start[d] / self.chunks[d]).collect();
        let chunk_end: Vec<u64> = (0..ndim)
            .map(|d| (end[d] + self.chunks[d] - 1) / self.chunks[d])
            .collect();

        let mut chunk_coords = chunk_start.clone();
        loop {
            // Read existing chunk (or fill)
            let mut chunk_data = self.read_chunk(&chunk_coords)?;

            let c_start: Vec<u64> = (0..ndim)
                .map(|d| chunk_coords[d] * self.chunks[d])
                .collect();
            let overlap_start: Vec<u64> = (0..ndim).map(|d| start[d].max(c_start[d])).collect();
            let overlap_end: Vec<u64> = (0..ndim)
                .map(|d| end[d].min(c_start[d] + self.chunks[d]))
                .collect();
            let overlap_shape: Vec<u64> = (0..ndim)
                .map(|d| overlap_end[d] - overlap_start[d])
                .collect();
            let overlap_total: usize = overlap_shape.iter().map(|&s| s as usize).product();

            for linear in 0..overlap_total {
                let mut multi = vec![0u64; ndim];
                let mut rem = linear;
                for d in (0..ndim).rev() {
                    multi[d] = (rem % overlap_shape[d] as usize) as u64;
                    rem /= overlap_shape[d] as usize;
                }

                let chunk_idx: Vec<u64> = (0..ndim)
                    .map(|d| overlap_start[d] + multi[d] - c_start[d])
                    .collect();
                let chunk_linear = c_order_index(&chunk_idx, &self.chunks);

                let sel_idx: Vec<u64> = (0..ndim)
                    .map(|d| overlap_start[d] + multi[d] - start[d])
                    .collect();
                let sel_linear = c_order_index(&sel_idx, &sel_shape);

                chunk_data[chunk_linear] = data[sel_linear];
            }

            self.write_chunk(&chunk_coords, &chunk_data)?;

            if !advance_coords(&mut chunk_coords, &chunk_start, &chunk_end, ndim) {
                break;
            }
        }

        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Compute C-order (row-major) linear index from multi-dim indices.
fn c_order_index(indices: &[u64], shape: &[u64]) -> usize {
    let mut linear = 0usize;
    let mut stride = 1usize;
    for d in (0..indices.len()).rev() {
        linear += indices[d] as usize * stride;
        stride *= shape[d] as usize;
    }
    linear
}

/// Advance multi-dimensional coordinates (last dim first). Returns false when done.
fn advance_coords(coords: &mut [u64], start: &[u64], end: &[u64], ndim: usize) -> bool {
    for d in (0..ndim).rev() {
        coords[d] += 1;
        if coords[d] < end[d] {
            return true;
        }
        coords[d] = start[d];
    }
    false
}

/// Build a codec pipeline for the given dtype and compression setting.
fn build_pipeline(dtype: DataType, compress: bool) -> CodecPipeline {
    let mut pipeline = CodecPipeline::new();
    pipeline.push(BytesCodec::new(Endian::Little, dtype.byte_size()));
    if compress {
        pipeline.push(ZstdCodec::default());
    }
    pipeline
}

/// Convert raw bytes to Vec<f64> according to the data type (little-endian).
fn bytes_to_f64(data: &[u8], dtype: DataType, n: usize) -> Result<Vec<f64>> {
    let elem = dtype.byte_size();
    if data.len() < n * elem {
        return Err(IoError::FormatError(format!(
            "Chunk data too short: {} bytes for {} elements of {} bytes each",
            data.len(),
            n,
            elem
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * elem;
        let val = match dtype {
            DataType::Bool => {
                if data[offset] != 0 {
                    1.0
                } else {
                    0.0
                }
            }
            DataType::Int8 => data[offset] as i8 as f64,
            DataType::Int16 => {
                let b = [data[offset], data[offset + 1]];
                i16::from_le_bytes(b) as f64
            }
            DataType::Int32 => {
                let mut b = [0u8; 4];
                b.copy_from_slice(&data[offset..offset + 4]);
                i32::from_le_bytes(b) as f64
            }
            DataType::Int64 => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&data[offset..offset + 8]);
                i64::from_le_bytes(b) as f64
            }
            DataType::UInt8 => data[offset] as f64,
            DataType::UInt16 => {
                let b = [data[offset], data[offset + 1]];
                u16::from_le_bytes(b) as f64
            }
            DataType::UInt32 => {
                let mut b = [0u8; 4];
                b.copy_from_slice(&data[offset..offset + 4]);
                u32::from_le_bytes(b) as f64
            }
            DataType::UInt64 => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&data[offset..offset + 8]);
                u64::from_le_bytes(b) as f64
            }
            DataType::Float32 => {
                let mut b = [0u8; 4];
                b.copy_from_slice(&data[offset..offset + 4]);
                f32::from_le_bytes(b) as f64
            }
            DataType::Float64 => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&data[offset..offset + 8]);
                f64::from_le_bytes(b)
            }
        };
        out.push(val);
    }
    Ok(out)
}

/// Convert Vec<f64> to raw bytes in the target data type (little-endian).
fn f64_to_bytes(data: &[f64], dtype: DataType) -> Result<Vec<u8>> {
    let elem = dtype.byte_size();
    let mut out = Vec::with_capacity(data.len() * elem);
    for &val in data {
        match dtype {
            DataType::Bool => out.push(if val != 0.0 { 1 } else { 0 }),
            DataType::Int8 => out.push(val as i8 as u8),
            DataType::Int16 => out.extend_from_slice(&(val as i16).to_le_bytes()),
            DataType::Int32 => out.extend_from_slice(&(val as i32).to_le_bytes()),
            DataType::Int64 => out.extend_from_slice(&(val as i64).to_le_bytes()),
            DataType::UInt8 => out.push(val as u8),
            DataType::UInt16 => out.extend_from_slice(&(val as u16).to_le_bytes()),
            DataType::UInt32 => out.extend_from_slice(&(val as u32).to_le_bytes()),
            DataType::UInt64 => out.extend_from_slice(&(val as u64).to_le_bytes()),
            DataType::Float32 => out.extend_from_slice(&(val as f32).to_le_bytes()),
            DataType::Float64 => out.extend_from_slice(&val.to_le_bytes()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_store(name: &str) -> (DirectoryStore, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("zarr_array_test_{name}"));
        let _ = fs::remove_dir_all(&dir);
        let store = DirectoryStore::open(&dir).expect("open store");
        (store, dir)
    }

    #[test]
    fn test_create_and_write_read_chunk_v2() {
        let (store, dir) = temp_store("v2_chunk");
        let arr = ZarrArray::create(
            store,
            "data",
            vec![10, 20],
            vec![5, 10],
            DataType::Float64,
            ZarrVersion::V2,
        )
        .expect("create");

        assert_eq!(arr.shape(), &[10, 20]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.num_chunks(), vec![2, 2]);

        // Write a chunk
        let chunk_data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        arr.write_chunk(&[0, 0], &chunk_data).expect("write chunk");

        // Read it back
        let read = arr.read_chunk(&[0, 0]).expect("read chunk");
        assert_eq!(read, chunk_data);

        // Read missing chunk -> fill values
        let fill = arr.read_chunk(&[1, 1]).expect("read missing");
        assert!(fill.iter().all(|&v| v == 0.0));
        assert_eq!(fill.len(), 50);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_create_and_write_read_chunk_v3() {
        let (store, dir) = temp_store("v3_chunk");
        let arr = ZarrArray::create(
            store,
            "arr3",
            vec![8],
            vec![4],
            DataType::Int32,
            ZarrVersion::V3,
        )
        .expect("create");

        let data: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
        arr.write_chunk(&[0], &data).expect("write");
        let read = arr.read_chunk(&[0]).expect("read");
        assert_eq!(read, data);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_open_existing_v2() {
        let (store, dir) = temp_store("v2_open");
        {
            let arr = ZarrArray::create(
                store,
                "myarr",
                vec![100],
                vec![25],
                DataType::Float32,
                ZarrVersion::V2,
            )
            .expect("create");
            let chunk: Vec<f64> = (0..25).map(|i| i as f64 * 0.5).collect();
            arr.write_chunk(&[2], &chunk).expect("write");
        }

        // Re-open
        let store2 = DirectoryStore::open(&dir).expect("reopen");
        let arr2 = ZarrArray::open(store2, "myarr").expect("open");
        assert_eq!(arr2.shape(), &[100]);
        assert_eq!(arr2.data_type(), DataType::Float32);
        let read = arr2.read_chunk(&[2]).expect("read");
        // Float32 roundtrip loses some precision
        assert_eq!(read.len(), 25);
        assert!((read[0] - 0.0).abs() < 1e-6);
        assert!((read[1] - 0.5).abs() < 1e-6);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_open_existing_v3() {
        let (store, dir) = temp_store("v3_open");
        {
            let arr = ZarrArray::create(
                store,
                "v3arr",
                vec![6, 4],
                vec![3, 2],
                DataType::Float64,
                ZarrVersion::V3,
            )
            .expect("create");
            let chunk: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            arr.write_chunk(&[0, 0], &chunk).expect("write");
        }

        let store2 = DirectoryStore::open(&dir).expect("reopen");
        let arr2 = ZarrArray::open(store2, "v3arr").expect("open");
        assert_eq!(arr2.shape(), &[6, 4]);
        assert_eq!(arr2.version(), ZarrVersion::V3);
        let read = arr2.read_chunk(&[0, 0]).expect("read");
        assert_eq!(read, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_slice_read_within_chunk() {
        let (store, dir) = temp_store("slice_within");
        let arr = ZarrArray::create(
            store,
            "s",
            vec![10],
            vec![5],
            DataType::Float64,
            ZarrVersion::V2,
        )
        .expect("create");

        let chunk0: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        arr.write_chunk(&[0], &chunk0).expect("write");

        let slice = arr.get(&[1], &[4]).expect("get");
        assert_eq!(slice, vec![11.0, 12.0, 13.0]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_slice_read_across_chunks() {
        let (store, dir) = temp_store("slice_across");
        let arr = ZarrArray::create(
            store,
            "x",
            vec![10],
            vec![4],
            DataType::Float64,
            ZarrVersion::V2,
        )
        .expect("create");

        // chunk 0: [0,1,2,3], chunk 1: [4,5,6,7], chunk 2: [8,9,fill,fill]
        arr.write_chunk(&[0], &[0.0, 1.0, 2.0, 3.0]).expect("w0");
        arr.write_chunk(&[1], &[4.0, 5.0, 6.0, 7.0]).expect("w1");
        arr.write_chunk(&[2], &[8.0, 9.0, 0.0, 0.0]).expect("w2");

        let slice = arr.get(&[2], &[7]).expect("get across");
        assert_eq!(slice, vec![2.0, 3.0, 4.0, 5.0, 6.0]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_slice_write_across_chunks() {
        let (store, dir) = temp_store("slice_write");
        let arr = ZarrArray::create_with_options(
            store,
            "w",
            vec![8],
            vec![4],
            DataType::Float64,
            ZarrVersion::V2,
            -1.0,
            false,
        )
        .expect("create");

        // Write across chunk boundary
        arr.set(&[2], &[6], &[10.0, 20.0, 30.0, 40.0]).expect("set");

        let all = arr.get(&[0], &[8]).expect("get all");
        assert_eq!(all, vec![-1.0, -1.0, 10.0, 20.0, 30.0, 40.0, -1.0, -1.0]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_2d_slice_across_chunks() {
        let (store, dir) = temp_store("2d_slice");
        let arr = ZarrArray::create(
            store,
            "m",
            vec![6, 6],
            vec![3, 3],
            DataType::Float64,
            ZarrVersion::V2,
        )
        .expect("create");

        // Fill chunk (0,0): 3x3
        let c00: Vec<f64> = (0..9).map(|i| (i + 1) as f64).collect();
        arr.write_chunk(&[0, 0], &c00).expect("write 0,0");

        // Fill chunk (0,1): 3x3
        let c01: Vec<f64> = (10..19).map(|i| i as f64).collect();
        arr.write_chunk(&[0, 1], &c01).expect("write 0,1");

        // Read across chunks: rows 0..2, cols 1..5
        let slice = arr.get(&[0, 1], &[2, 5]).expect("get 2d");
        // Expected from C-order:
        // row0: c00[0,1], c00[0,2], c01[0,0], c01[0,1] = 2,3,10,11
        // row1: c00[1,1], c00[1,2], c01[1,0], c01[1,1] = 5,6,13,14
        assert_eq!(slice, vec![2.0, 3.0, 10.0, 11.0, 5.0, 6.0, 13.0, 14.0]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_uint8_dtype() {
        let (store, dir) = temp_store("uint8");
        let arr = ZarrArray::create(
            store,
            "u8",
            vec![4],
            vec![4],
            DataType::UInt8,
            ZarrVersion::V2,
        )
        .expect("create");

        arr.write_chunk(&[0], &[0.0, 127.0, 200.0, 255.0])
            .expect("write");
        let read = arr.read_chunk(&[0]).expect("read");
        assert_eq!(read, vec![0.0, 127.0, 200.0, 255.0]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_fill_value_custom() {
        let (store, dir) = temp_store("fill_custom");
        let arr = ZarrArray::create_with_options(
            store,
            "f",
            vec![10],
            vec![5],
            DataType::Float64,
            ZarrVersion::V2,
            f64::NAN,
            false,
        )
        .expect("create");

        let read = arr.read_chunk(&[0]).expect("read missing");
        assert!(read.iter().all(|v| v.is_nan()));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_uncompressed_array() {
        let (store, dir) = temp_store("no_compress");
        let arr = ZarrArray::create_with_options(
            store,
            "nc",
            vec![4],
            vec![4],
            DataType::Float64,
            ZarrVersion::V2,
            0.0,
            false,
        )
        .expect("create");

        let data = vec![1.0, 2.0, 3.0, 4.0];
        arr.write_chunk(&[0], &data).expect("write");
        let read = arr.read_chunk(&[0]).expect("read");
        assert_eq!(read, data);

        let _ = fs::remove_dir_all(&dir);
    }
}
