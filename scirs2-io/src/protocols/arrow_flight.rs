//! Arrow Flight protocol — pure-Rust in-memory simulation.
//!
//! Implements:
//! - Arrow IPC message framing (simplified, little-endian raw column bytes)
//! - [`InMemoryFlightServer`]: register datasets, list flights, do-get/do-put
//! - [`ArrowFlightClient`]: thin wrapper that delegates to the server
//!
//! No external network I/O or gRPC is required; all communication is in-process.
//!
//! # Example
//! ```rust
//! use scirs2_io::protocols::arrow_flight::*;
//!
//! let mut server = InMemoryFlightServer::new();
//! let schema = ArrowSchema {
//!     fields: vec![SchemaField { name: "x".into(), data_type: ArrowDataType::Float64, nullable: false }],
//! };
//! let batch = RecordBatch {
//!     schema: schema.clone(),
//!     columns: vec![ColumnData::Float64(vec![1.0, 2.0, 3.0])],
//!     num_rows: 3,
//! };
//! server.register_dataset("demo", vec![batch]);
//!
//! let client = ArrowFlightClient::new(&server);
//! let flights = client.list_flights();
//! assert_eq!(flights.len(), 1);
//! ```

use std::collections::HashMap;

use crate::error::{IoError, Result as IoResult};

// ─────────────────────────────────── types ───────────────────────────────────

/// Arrow IPC message type tag.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// Schema message — describes field names and types.
    Schema,
    /// RecordBatch message — column buffers for a slice of rows.
    RecordBatch,
    /// DictionaryBatch message — encoded dictionary values.
    DictionaryBatch,
}

/// Arrow logical data type (simplified subset).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrowDataType {
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit IEEE float.
    Float32,
    /// 64-bit IEEE float.
    Float64,
    /// Variable-length UTF-8 string (offsets + data buffers).
    Utf8,
    /// Boolean (bit-packed).
    Boolean,
    /// Variable-length list of a child type.
    List(Box<ArrowDataType>),
}

impl ArrowDataType {
    /// Encode this type to a one-byte tag for serialization.
    fn type_tag(&self) -> u8 {
        match self {
            ArrowDataType::Int32 => 1,
            ArrowDataType::Int64 => 2,
            ArrowDataType::Float32 => 3,
            ArrowDataType::Float64 => 4,
            ArrowDataType::Utf8 => 5,
            ArrowDataType::Boolean => 6,
            ArrowDataType::List(_) => 7,
        }
    }

    /// Decode from a one-byte tag plus optional child bytes.
    fn from_tag(tag: u8, child: Option<ArrowDataType>) -> IoResult<ArrowDataType> {
        match tag {
            1 => Ok(ArrowDataType::Int32),
            2 => Ok(ArrowDataType::Int64),
            3 => Ok(ArrowDataType::Float32),
            4 => Ok(ArrowDataType::Float64),
            5 => Ok(ArrowDataType::Utf8),
            6 => Ok(ArrowDataType::Boolean),
            7 => {
                let inner = child.ok_or_else(|| {
                    IoError::FormatError("List type missing child type".into())
                })?;
                Ok(ArrowDataType::List(Box::new(inner)))
            }
            other => Err(IoError::FormatError(format!(
                "Unknown ArrowDataType tag: {other}"
            ))),
        }
    }
}

/// A single field in an Arrow schema.
#[derive(Debug, Clone)]
pub struct SchemaField {
    /// Field name.
    pub name: String,
    /// Logical data type.
    pub data_type: ArrowDataType,
    /// Whether the column may contain nulls.
    pub nullable: bool,
}

/// An Arrow schema: ordered list of fields.
#[derive(Debug, Clone)]
pub struct ArrowSchema {
    /// Ordered list of fields.
    pub fields: Vec<SchemaField>,
}

/// A columnar record batch (a slice of rows across multiple typed columns).
#[derive(Debug, Clone)]
pub struct RecordBatch {
    /// Schema describing column types.
    pub schema: ArrowSchema,
    /// One entry per schema field, in the same order.
    pub columns: Vec<ColumnData>,
    /// Number of logical rows.
    pub num_rows: usize,
}

/// Typed, owned column data.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ColumnData {
    /// 32-bit signed integer column.
    Int32(Vec<i32>),
    /// 64-bit signed integer column.
    Int64(Vec<i64>),
    /// 64-bit float column.
    Float64(Vec<f64>),
    /// 32-bit float column.
    Float32(Vec<f32>),
    /// UTF-8 string column.
    Utf8(Vec<String>),
    /// Boolean column.
    Boolean(Vec<bool>),
}

impl ColumnData {
    /// Number of elements in this column.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int32(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::Float32(v) => v.len(),
            ColumnData::Utf8(v) => v.len(),
            ColumnData::Boolean(v) => v.len(),
        }
    }

    /// Returns `true` if this column contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The logical Arrow data type for this column.
    pub fn data_type(&self) -> ArrowDataType {
        match self {
            ColumnData::Int32(_) => ArrowDataType::Int32,
            ColumnData::Int64(_) => ArrowDataType::Int64,
            ColumnData::Float64(_) => ArrowDataType::Float64,
            ColumnData::Float32(_) => ArrowDataType::Float32,
            ColumnData::Utf8(_) => ArrowDataType::Utf8,
            ColumnData::Boolean(_) => ArrowDataType::Boolean,
        }
    }
}

// ─────────────────────────────── IPC framing ─────────────────────────────────

/// Magic bytes placed at the start of every IPC message.
/// In the real Arrow IPC spec this is `[0xFF, 0xFF, 0xFF, 0xFF]` (continuation marker).
const IPC_CONTINUATION: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
/// Protocol version tag embedded in messages.
const IPC_VERSION: u8 = 1;

// ── Schema serialization ──────────────────────────────────────────────────────

/// Encode a data type recursively.
///
/// Format: `[tag: u8]`; for `List`, `[7, child_bytes...]`.
fn encode_data_type(dt: &ArrowDataType, buf: &mut Vec<u8>) {
    match dt {
        ArrowDataType::List(child) => {
            buf.push(7);
            encode_data_type(child, buf);
        }
        other => buf.push(other.type_tag()),
    }
}

/// Decode a data type starting at `offset`; returns `(type, new_offset)`.
fn decode_data_type(data: &[u8], offset: usize) -> IoResult<(ArrowDataType, usize)> {
    if offset >= data.len() {
        return Err(IoError::FormatError(
            "ArrowDataType: unexpected end of data".into(),
        ));
    }
    let tag = data[offset];
    if tag == 7 {
        let (child, next) = decode_data_type(data, offset + 1)?;
        Ok((ArrowDataType::List(Box::new(child)), next))
    } else {
        let dt = ArrowDataType::from_tag(tag, None)?;
        Ok((dt, offset + 1))
    }
}

/// Serialize an [`ArrowSchema`] to bytes.
///
/// Layout:
/// ```text
/// [version: u8]
/// [n_fields: u32 LE]
/// for each field:
///   [name_len: u16 LE] [name bytes: UTF-8]
///   [data_type bytes (recursive)]
///   [nullable: u8]
/// ```
pub fn encode_schema(schema: &ArrowSchema) -> IoResult<Vec<u8>> {
    let mut buf = Vec::new();
    buf.push(IPC_VERSION);
    buf.extend_from_slice(&(schema.fields.len() as u32).to_le_bytes());
    for field in &schema.fields {
        let name_bytes = field.name.as_bytes();
        let name_len = name_bytes.len() as u16;
        buf.extend_from_slice(&name_len.to_le_bytes());
        buf.extend_from_slice(name_bytes);
        encode_data_type(&field.data_type, &mut buf);
        buf.push(if field.nullable { 1 } else { 0 });
    }
    Ok(buf)
}

/// Deserialize an [`ArrowSchema`] from bytes produced by [`encode_schema`].
pub fn decode_schema(data: &[u8]) -> IoResult<ArrowSchema> {
    if data.is_empty() {
        return Err(IoError::FormatError("Empty schema bytes".into()));
    }
    let version = data[0];
    if version != IPC_VERSION {
        return Err(IoError::FormatError(format!(
            "Unknown IPC version: {version}"
        )));
    }
    if data.len() < 5 {
        return Err(IoError::FormatError("Schema header too short".into()));
    }
    let n_fields = u32::from_le_bytes(
        data[1..5]
            .try_into()
            .map_err(|_| IoError::FormatError("Schema n_fields: bad bytes".into()))?,
    ) as usize;

    let mut offset = 5;
    let mut fields = Vec::with_capacity(n_fields);

    for _ in 0..n_fields {
        if offset + 2 > data.len() {
            return Err(IoError::FormatError("Schema field name_len: truncated".into()));
        }
        let name_len = u16::from_le_bytes(
            data[offset..offset + 2]
                .try_into()
                .map_err(|_| IoError::FormatError("Schema name_len bytes".into()))?,
        ) as usize;
        offset += 2;

        if offset + name_len > data.len() {
            return Err(IoError::FormatError("Schema field name: truncated".into()));
        }
        let name = std::str::from_utf8(&data[offset..offset + name_len])
            .map_err(|e| IoError::FormatError(format!("Schema field name UTF-8: {e}")))?
            .to_string();
        offset += name_len;

        let (data_type, next) = decode_data_type(data, offset)?;
        offset = next;

        if offset >= data.len() {
            return Err(IoError::FormatError("Schema field nullable: truncated".into()));
        }
        let nullable = data[offset] != 0;
        offset += 1;

        fields.push(SchemaField {
            name,
            data_type,
            nullable,
        });
    }

    Ok(ArrowSchema { fields })
}

// ── Column serialization ──────────────────────────────────────────────────────

/// Serialize a single [`ColumnData`] to raw bytes.
///
/// Layout:
/// ```text
/// [type_tag: u8]
/// [n_values: u64 LE]
/// [raw value bytes: little-endian, fixed or variable]
/// ```
/// For `Utf8`: `n_values u64`, then for each string `[len: u32 LE][bytes]`.
/// For `Boolean`: packed as one `u8` per element (0/1) for simplicity.
fn encode_column(col: &ColumnData) -> Vec<u8> {
    let mut buf = Vec::new();
    match col {
        ColumnData::Int32(v) => {
            buf.push(1u8);
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::Int64(v) => {
            buf.push(2u8);
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::Float32(v) => {
            buf.push(3u8);
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::Float64(v) => {
            buf.push(4u8);
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        ColumnData::Utf8(v) => {
            buf.push(5u8);
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for s in v {
                let sb = s.as_bytes();
                buf.extend_from_slice(&(sb.len() as u32).to_le_bytes());
                buf.extend_from_slice(sb);
            }
        }
        ColumnData::Boolean(v) => {
            buf.push(6u8);
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for &b in v {
                buf.push(if b { 1 } else { 0 });
            }
        }
    }
    buf
}

/// Decode a [`ColumnData`] from the raw bytes written by [`encode_column`].
/// Returns `(column, bytes_consumed)`.
fn decode_column(data: &[u8], offset: usize) -> IoResult<(ColumnData, usize)> {
    if offset >= data.len() {
        return Err(IoError::FormatError("Column: unexpected end".into()));
    }
    let tag = data[offset];
    let mut pos = offset + 1;

    if pos + 8 > data.len() {
        return Err(IoError::FormatError("Column: n_values truncated".into()));
    }
    let n = u64::from_le_bytes(
        data[pos..pos + 8]
            .try_into()
            .map_err(|_| IoError::FormatError("Column n_values bytes".into()))?,
    ) as usize;
    pos += 8;

    match tag {
        1 => {
            // Int32
            let needed = n * 4;
            if pos + needed > data.len() {
                return Err(IoError::FormatError("Int32 column: truncated".into()));
            }
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let s = pos + i * 4;
                let x = i32::from_le_bytes(
                    data[s..s + 4]
                        .try_into()
                        .map_err(|_| IoError::FormatError("Int32 element".into()))?,
                );
                v.push(x);
            }
            Ok((ColumnData::Int32(v), pos + needed))
        }
        2 => {
            // Int64
            let needed = n * 8;
            if pos + needed > data.len() {
                return Err(IoError::FormatError("Int64 column: truncated".into()));
            }
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let s = pos + i * 8;
                let x = i64::from_le_bytes(
                    data[s..s + 8]
                        .try_into()
                        .map_err(|_| IoError::FormatError("Int64 element".into()))?,
                );
                v.push(x);
            }
            Ok((ColumnData::Int64(v), pos + needed))
        }
        3 => {
            // Float32
            let needed = n * 4;
            if pos + needed > data.len() {
                return Err(IoError::FormatError("Float32 column: truncated".into()));
            }
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let s = pos + i * 4;
                let x = f32::from_le_bytes(
                    data[s..s + 4]
                        .try_into()
                        .map_err(|_| IoError::FormatError("Float32 element".into()))?,
                );
                v.push(x);
            }
            Ok((ColumnData::Float32(v), pos + needed))
        }
        4 => {
            // Float64
            let needed = n * 8;
            if pos + needed > data.len() {
                return Err(IoError::FormatError("Float64 column: truncated".into()));
            }
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let s = pos + i * 8;
                let x = f64::from_le_bytes(
                    data[s..s + 8]
                        .try_into()
                        .map_err(|_| IoError::FormatError("Float64 element".into()))?,
                );
                v.push(x);
            }
            Ok((ColumnData::Float64(v), pos + needed))
        }
        5 => {
            // Utf8
            let mut strings = Vec::with_capacity(n);
            for _ in 0..n {
                if pos + 4 > data.len() {
                    return Err(IoError::FormatError("Utf8 column: length truncated".into()));
                }
                let slen = u32::from_le_bytes(
                    data[pos..pos + 4]
                        .try_into()
                        .map_err(|_| IoError::FormatError("Utf8 str len".into()))?,
                ) as usize;
                pos += 4;
                if pos + slen > data.len() {
                    return Err(IoError::FormatError("Utf8 column: string truncated".into()));
                }
                let s = std::str::from_utf8(&data[pos..pos + slen])
                    .map_err(|e| IoError::FormatError(format!("Utf8 column UTF-8: {e}")))?
                    .to_string();
                pos += slen;
                strings.push(s);
            }
            Ok((ColumnData::Utf8(strings), pos))
        }
        6 => {
            // Boolean
            if pos + n > data.len() {
                return Err(IoError::FormatError("Boolean column: truncated".into()));
            }
            let v: Vec<bool> = data[pos..pos + n].iter().map(|&b| b != 0).collect();
            Ok((ColumnData::Boolean(v), pos + n))
        }
        other => Err(IoError::FormatError(format!(
            "Unknown column type tag: {other}"
        ))),
    }
}

// ── RecordBatch framing ───────────────────────────────────────────────────────

/// Encode a [`RecordBatch`] into Arrow IPC framing bytes.
///
/// Frame layout:
/// ```text
/// [continuation: 4 bytes = 0xFFFFFFFF]
/// [metadata_len: 4 bytes LE]
/// [metadata: schema_bytes (metadata_len bytes)]
/// [body_len: 8 bytes LE]
/// [num_rows: 8 bytes LE]
/// [n_columns: 4 bytes LE]
/// for each column:
///   [col_len: 8 bytes LE]
///   [column bytes (col_len bytes)]
/// ```
pub fn encode_record_batch(batch: &RecordBatch) -> IoResult<Vec<u8>> {
    // Validate column count matches schema
    if batch.columns.len() != batch.schema.fields.len() {
        return Err(IoError::FormatError(format!(
            "RecordBatch column count {} != schema field count {}",
            batch.columns.len(),
            batch.schema.fields.len()
        )));
    }

    let schema_bytes = encode_schema(&batch.schema)?;
    let metadata_len = schema_bytes.len() as u32;

    // Encode each column
    let col_bytes: Vec<Vec<u8>> = batch.columns.iter().map(encode_column).collect();

    let mut buf = Vec::new();
    // Continuation marker
    buf.extend_from_slice(&IPC_CONTINUATION);
    // Metadata (schema) length
    buf.extend_from_slice(&metadata_len.to_le_bytes());
    // Schema bytes
    buf.extend_from_slice(&schema_bytes);
    // Total body length (placeholder-style: num_rows + n_cols + column data)
    let body_len: u64 = 8 + 4 + col_bytes.iter().map(|c| 8 + c.len() as u64).sum::<u64>();
    buf.extend_from_slice(&body_len.to_le_bytes());
    // Number of rows
    buf.extend_from_slice(&(batch.num_rows as u64).to_le_bytes());
    // Number of columns
    buf.extend_from_slice(&(col_bytes.len() as u32).to_le_bytes());
    // Each column
    for cb in &col_bytes {
        buf.extend_from_slice(&(cb.len() as u64).to_le_bytes());
        buf.extend_from_slice(cb);
    }

    Ok(buf)
}

/// Decode a [`RecordBatch`] from bytes produced by [`encode_record_batch`].
pub fn decode_record_batch(data: &[u8]) -> IoResult<RecordBatch> {
    // Verify continuation marker
    if data.len() < 4 {
        return Err(IoError::FormatError(
            "RecordBatch IPC: too short for continuation".into(),
        ));
    }
    if data[0..4] != IPC_CONTINUATION {
        return Err(IoError::FormatError(
            "RecordBatch IPC: bad continuation marker".into(),
        ));
    }
    let mut pos = 4;

    // Metadata length
    if pos + 4 > data.len() {
        return Err(IoError::FormatError(
            "RecordBatch IPC: metadata_len truncated".into(),
        ));
    }
    let metadata_len = u32::from_le_bytes(
        data[pos..pos + 4]
            .try_into()
            .map_err(|_| IoError::FormatError("metadata_len bytes".into()))?,
    ) as usize;
    pos += 4;

    // Schema
    if pos + metadata_len > data.len() {
        return Err(IoError::FormatError(
            "RecordBatch IPC: schema bytes truncated".into(),
        ));
    }
    let schema = decode_schema(&data[pos..pos + metadata_len])?;
    pos += metadata_len;

    // Body length (skip, we use the encoded lengths instead)
    if pos + 8 > data.len() {
        return Err(IoError::FormatError(
            "RecordBatch IPC: body_len truncated".into(),
        ));
    }
    pos += 8; // skip body_len field

    // num_rows
    if pos + 8 > data.len() {
        return Err(IoError::FormatError(
            "RecordBatch IPC: num_rows truncated".into(),
        ));
    }
    let num_rows = u64::from_le_bytes(
        data[pos..pos + 8]
            .try_into()
            .map_err(|_| IoError::FormatError("num_rows bytes".into()))?,
    ) as usize;
    pos += 8;

    // n_columns
    if pos + 4 > data.len() {
        return Err(IoError::FormatError(
            "RecordBatch IPC: n_columns truncated".into(),
        ));
    }
    let n_columns = u32::from_le_bytes(
        data[pos..pos + 4]
            .try_into()
            .map_err(|_| IoError::FormatError("n_columns bytes".into()))?,
    ) as usize;
    pos += 4;

    // Columns
    let mut columns = Vec::with_capacity(n_columns);
    for _ in 0..n_columns {
        if pos + 8 > data.len() {
            return Err(IoError::FormatError(
                "RecordBatch IPC: col_len truncated".into(),
            ));
        }
        let col_len = u64::from_le_bytes(
            data[pos..pos + 8]
                .try_into()
                .map_err(|_| IoError::FormatError("col_len bytes".into()))?,
        ) as usize;
        pos += 8;

        if pos + col_len > data.len() {
            return Err(IoError::FormatError(
                "RecordBatch IPC: column data truncated".into(),
            ));
        }
        let (col, _) = decode_column(&data[pos..pos + col_len], 0)?;
        columns.push(col);
        pos += col_len;
    }

    Ok(RecordBatch {
        schema,
        columns,
        num_rows,
    })
}

// ─────────────────────────────── Flight types ────────────────────────────────

/// Identifies a data stream within a Flight service.
#[derive(Debug, Clone)]
pub struct FlightDescriptor {
    /// What kind of descriptor this is.
    pub kind: FlightDescriptorKind,
}

/// Variant of a [`FlightDescriptor`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum FlightDescriptorKind {
    /// Path-based descriptor: a list of string path segments.
    Path(Vec<String>),
    /// Command-based descriptor: an opaque byte command.
    Cmd(Vec<u8>),
}

impl FlightDescriptor {
    /// Convenience constructor for a single-segment path descriptor.
    pub fn path(p: &str) -> Self {
        FlightDescriptor {
            kind: FlightDescriptorKind::Path(vec![p.to_string()]),
        }
    }

    /// Extract the primary path string, if this is a `Path` descriptor.
    pub fn primary_path(&self) -> Option<&str> {
        match &self.kind {
            FlightDescriptorKind::Path(segs) => segs.first().map(|s| s.as_str()),
            FlightDescriptorKind::Cmd(_) => None,
        }
    }
}

/// Endpoint information: how to retrieve a stream given a ticket.
#[derive(Debug, Clone)]
pub struct FlightEndpoint {
    /// Opaque ticket bytes identifying the stream shard.
    pub ticket: Vec<u8>,
    /// URIs of servers that can serve this endpoint (empty = this server).
    pub locations: Vec<String>,
}

/// Metadata describing an available data stream.
#[derive(Debug, Clone)]
pub struct FlightInfo {
    /// Encoded schema bytes (as from [`encode_schema`]).
    pub schema_bytes: Vec<u8>,
    /// One or more endpoints to retrieve the data from.
    pub endpoints: Vec<FlightEndpoint>,
    /// Total number of records across all endpoints (−1 if unknown).
    pub total_records: i64,
    /// Total size in bytes across all endpoints (−1 if unknown).
    pub total_bytes: i64,
    /// The descriptor that identifies this stream.
    pub descriptor: FlightDescriptor,
}

// ─────────────────────────────── Server ──────────────────────────────────────

/// In-memory Arrow Flight server for testing and local I/O.
///
/// Stores datasets keyed by path string. Implements the core Flight RPC
/// methods without any real network transport.
pub struct InMemoryFlightServer {
    data_store: HashMap<String, Vec<RecordBatch>>,
}

impl Default for InMemoryFlightServer {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryFlightServer {
    /// Create a new, empty server.
    pub fn new() -> Self {
        InMemoryFlightServer {
            data_store: HashMap::new(),
        }
    }

    /// Register a dataset under the given path name.
    pub fn register_dataset(&mut self, path: &str, batches: Vec<RecordBatch>) {
        self.data_store.insert(path.to_string(), batches);
    }

    /// List metadata for all registered datasets (equivalent to `ListFlights`).
    pub fn list_flights(&self) -> Vec<FlightInfo> {
        self.data_store
            .iter()
            .filter_map(|(path, batches)| self.build_flight_info(path, batches).ok())
            .collect()
    }

    /// Retrieve metadata for the dataset identified by `descriptor` (`GetFlightInfo`).
    pub fn get_flight_info(&self, descriptor: &FlightDescriptor) -> IoResult<FlightInfo> {
        let path = descriptor
            .primary_path()
            .ok_or_else(|| IoError::NotFound("FlightDescriptor has no path".into()))?;
        let batches = self
            .data_store
            .get(path)
            .ok_or_else(|| IoError::NotFound(format!("No dataset at path '{path}'")))?;
        self.build_flight_info(path, batches)
    }

    /// Retrieve all [`RecordBatch`]es for a ticket (ticket = path as UTF-8 bytes).
    pub fn do_get(&self, ticket: &[u8]) -> IoResult<Vec<RecordBatch>> {
        let path = std::str::from_utf8(ticket)
            .map_err(|e| IoError::FormatError(format!("Ticket UTF-8: {e}")))?;
        let batches = self
            .data_store
            .get(path)
            .ok_or_else(|| IoError::NotFound(format!("No dataset for ticket '{path}'")))?;
        Ok(batches.clone())
    }

    /// Upload batches for a descriptor (equivalent to `DoPut`).
    pub fn do_put(
        &mut self,
        descriptor: &FlightDescriptor,
        batches: Vec<RecordBatch>,
    ) -> IoResult<()> {
        let path = descriptor
            .primary_path()
            .ok_or_else(|| IoError::NotFound("FlightDescriptor has no path".into()))?
            .to_string();
        self.data_store.insert(path, batches);
        Ok(())
    }

    /// Return the schema for the dataset identified by `descriptor`.
    pub fn get_schema(&self, descriptor: &FlightDescriptor) -> IoResult<ArrowSchema> {
        let path = descriptor
            .primary_path()
            .ok_or_else(|| IoError::NotFound("FlightDescriptor has no path".into()))?;
        let batches = self
            .data_store
            .get(path)
            .ok_or_else(|| IoError::NotFound(format!("No dataset at path '{path}'")))?;
        let first = batches
            .first()
            .ok_or_else(|| IoError::NotFound("Dataset has no batches".into()))?;
        Ok(first.schema.clone())
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn build_flight_info(&self, path: &str, batches: &[RecordBatch]) -> IoResult<FlightInfo> {
        let schema_bytes = if let Some(first) = batches.first() {
            encode_schema(&first.schema)?
        } else {
            encode_schema(&ArrowSchema { fields: vec![] })?
        };

        let total_records: i64 = batches.iter().map(|b| b.num_rows as i64).sum();

        let ticket = path.as_bytes().to_vec();
        let endpoints = vec![FlightEndpoint {
            ticket,
            locations: vec![],
        }];

        Ok(FlightInfo {
            schema_bytes,
            endpoints,
            total_records,
            total_bytes: -1,
            descriptor: FlightDescriptor::path(path),
        })
    }
}

// ─────────────────────────────── Client ──────────────────────────────────────

/// Arrow Flight client backed by an [`InMemoryFlightServer`].
///
/// In a real deployment this would open a TCP connection and speak the
/// Arrow Flight gRPC wire protocol. Here it delegates directly to the server.
pub struct ArrowFlightClient<'a> {
    server: &'a InMemoryFlightServer,
}

impl<'a> ArrowFlightClient<'a> {
    /// Wrap a reference to an existing in-memory server.
    pub fn new(server: &'a InMemoryFlightServer) -> Self {
        ArrowFlightClient { server }
    }

    /// List all available flights.
    pub fn list_flights(&self) -> Vec<FlightInfo> {
        self.server.list_flights()
    }

    /// Get metadata for a specific flight.
    pub fn get_flight_info(&self, descriptor: &FlightDescriptor) -> IoResult<FlightInfo> {
        self.server.get_flight_info(descriptor)
    }

    /// Download all record batches for the given ticket.
    pub fn do_get(&self, ticket: &[u8]) -> IoResult<Vec<RecordBatch>> {
        self.server.do_get(ticket)
    }

    /// Get the schema for a flight.
    pub fn get_schema(&self, descriptor: &FlightDescriptor) -> IoResult<ArrowSchema> {
        self.server.get_schema(descriptor)
    }
}

// ─────────────────────────────────── tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_schema() -> ArrowSchema {
        ArrowSchema {
            fields: vec![
                SchemaField {
                    name: "id".into(),
                    data_type: ArrowDataType::Int32,
                    nullable: false,
                },
                SchemaField {
                    name: "value".into(),
                    data_type: ArrowDataType::Float64,
                    nullable: true,
                },
                SchemaField {
                    name: "label".into(),
                    data_type: ArrowDataType::Utf8,
                    nullable: true,
                },
            ],
        }
    }

    fn make_simple_batch() -> RecordBatch {
        let schema = make_simple_schema();
        RecordBatch {
            schema: schema.clone(),
            columns: vec![
                ColumnData::Int32(vec![1, 2, 3]),
                ColumnData::Float64(vec![1.1, 2.2, 3.3]),
                ColumnData::Utf8(vec!["a".into(), "b".into(), "c".into()]),
            ],
            num_rows: 3,
        }
    }

    #[test]
    fn test_encode_decode_record_batch_roundtrip() {
        let batch = make_simple_batch();
        let encoded = encode_record_batch(&batch).expect("encode");
        let decoded = decode_record_batch(&encoded).expect("decode");

        assert_eq!(decoded.num_rows, batch.num_rows);
        assert_eq!(decoded.columns.len(), batch.columns.len());

        // Verify Int32 column
        if let (ColumnData::Int32(orig), ColumnData::Int32(dec)) =
            (&batch.columns[0], &decoded.columns[0])
        {
            assert_eq!(orig, dec);
        } else {
            panic!("Column 0 type mismatch");
        }

        // Verify Float64 column
        if let (ColumnData::Float64(orig), ColumnData::Float64(dec)) =
            (&batch.columns[1], &decoded.columns[1])
        {
            for (a, b) in orig.iter().zip(dec.iter()) {
                assert!((a - b).abs() < f64::EPSILON);
            }
        } else {
            panic!("Column 1 type mismatch");
        }

        // Verify Utf8 column
        if let (ColumnData::Utf8(orig), ColumnData::Utf8(dec)) =
            (&batch.columns[2], &decoded.columns[2])
        {
            assert_eq!(orig, dec);
        } else {
            panic!("Column 2 type mismatch");
        }
    }

    #[test]
    fn test_schema_field_names_preserved() {
        let schema = make_simple_schema();
        let encoded = encode_schema(&schema).expect("encode schema");
        let decoded = decode_schema(&encoded).expect("decode schema");

        let names: Vec<&str> = decoded.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["id", "value", "label"]);
    }

    #[test]
    fn test_record_batch_num_rows() {
        let batch = make_simple_batch();
        let encoded = encode_record_batch(&batch).expect("encode");
        let decoded = decode_record_batch(&encoded).expect("decode");
        assert_eq!(decoded.num_rows, 3);
    }

    #[test]
    fn test_flight_server_register_and_list() {
        let mut server = InMemoryFlightServer::new();
        server.register_dataset("ds1", vec![make_simple_batch()]);
        server.register_dataset("ds2", vec![make_simple_batch()]);

        let flights = server.list_flights();
        assert_eq!(flights.len(), 2);
    }

    #[test]
    fn test_flight_server_do_get_returns_batches() {
        let mut server = InMemoryFlightServer::new();
        let batch = make_simple_batch();
        server.register_dataset("test_path", vec![batch.clone()]);

        let retrieved = server.do_get(b"test_path").expect("do_get");
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].num_rows, 3);
    }

    #[test]
    fn test_flight_server_do_put_stores_batches() {
        let mut server = InMemoryFlightServer::new();
        let descriptor = FlightDescriptor::path("uploaded");
        let batch = make_simple_batch();

        server
            .do_put(&descriptor, vec![batch])
            .expect("do_put");

        let retrieved = server.do_get(b"uploaded").expect("do_get after do_put");
        assert_eq!(retrieved.len(), 1);
    }

    #[test]
    fn test_flight_client_list_flights() {
        let mut server = InMemoryFlightServer::new();
        server.register_dataset("alpha", vec![make_simple_batch()]);
        server.register_dataset("beta", vec![make_simple_batch()]);

        let client = ArrowFlightClient::new(&server);
        let flights = client.list_flights();
        assert_eq!(flights.len(), 2);
    }

    #[test]
    fn test_flight_client_do_get() {
        let mut server = InMemoryFlightServer::new();
        server.register_dataset("gamma", vec![make_simple_batch()]);

        let client = ArrowFlightClient::new(&server);
        let batches = client.do_get(b"gamma").expect("client do_get");
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows, 3);
    }

    #[test]
    fn test_flight_descriptor_path() {
        let desc = FlightDescriptor::path("my/dataset");
        assert_eq!(desc.primary_path(), Some("my/dataset"));

        match &desc.kind {
            FlightDescriptorKind::Path(segs) => assert_eq!(segs[0], "my/dataset"),
            _ => panic!("expected Path kind"),
        }
    }

    #[test]
    fn test_arrow_data_type_roundtrip() {
        let types = vec![
            ArrowDataType::Int32,
            ArrowDataType::Int64,
            ArrowDataType::Float32,
            ArrowDataType::Float64,
            ArrowDataType::Utf8,
            ArrowDataType::Boolean,
            ArrowDataType::List(Box::new(ArrowDataType::Float64)),
        ];

        for dt in &types {
            let mut buf = Vec::new();
            encode_data_type(dt, &mut buf);
            let (decoded, consumed) = decode_data_type(&buf, 0).expect("decode");
            assert_eq!(&decoded, dt);
            assert_eq!(consumed, buf.len());
        }
    }

    #[test]
    fn test_boolean_column_roundtrip() {
        let schema = ArrowSchema {
            fields: vec![SchemaField {
                name: "flags".into(),
                data_type: ArrowDataType::Boolean,
                nullable: false,
            }],
        };
        let batch = RecordBatch {
            schema,
            columns: vec![ColumnData::Boolean(vec![true, false, true, true, false])],
            num_rows: 5,
        };
        let enc = encode_record_batch(&batch).expect("encode");
        let dec = decode_record_batch(&enc).expect("decode");
        if let ColumnData::Boolean(v) = &dec.columns[0] {
            assert_eq!(v, &vec![true, false, true, true, false]);
        } else {
            panic!("Boolean column type mismatch");
        }
    }

    #[test]
    fn test_empty_batch_roundtrip() {
        let schema = ArrowSchema { fields: vec![] };
        let batch = RecordBatch {
            schema,
            columns: vec![],
            num_rows: 0,
        };
        let enc = encode_record_batch(&batch).expect("encode");
        let dec = decode_record_batch(&enc).expect("decode");
        assert_eq!(dec.num_rows, 0);
        assert_eq!(dec.columns.len(), 0);
    }
}
