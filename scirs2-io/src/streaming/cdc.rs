//! Change Data Capture (CDC) streaming support
//!
//! Provides ordered change event tracking with Log Sequence Numbers (LSN),
//! Debezium-format JSON parsing, CDC log replay for table-state reconstruction,
//! and materialisation of CDC streams to columnar (Parquet-lite) format.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::error::{IoError, Result};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A single cell value in a row image.
#[derive(Debug, Clone, PartialEq)]
pub enum CdcValue {
    /// NULL / absent value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// 64-bit integer value.
    Int(i64),
    /// 64-bit floating-point value.
    Float(f64),
    /// UTF-8 text value.
    Text(String),
    /// Raw byte sequence.
    Bytes(Vec<u8>),
}

impl CdcValue {
    /// Return a human-readable display string.
    pub fn to_display(&self) -> String {
        match self {
            CdcValue::Null => "NULL".to_string(),
            CdcValue::Bool(b) => b.to_string(),
            CdcValue::Int(i) => i.to_string(),
            CdcValue::Float(f) => f.to_string(),
            CdcValue::Text(s) => s.clone(),
            CdcValue::Bytes(b) => format!("<bytes len={}>", b.len()),
        }
    }
}

/// A row image — an ordered map of column name → value.
pub type RowImage = HashMap<String, CdcValue>;

/// The operation that generated a CDC event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CdcOperation {
    /// A row was inserted.
    Insert,
    /// A row was updated.
    Update,
    /// A row was deleted.
    Delete,
    /// A snapshot read (no previous state, used during initial load).
    Read,
    /// Schema-change DDL event.
    SchemaChange,
}

impl CdcOperation {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "c" | "insert" | "INSERT" | "i" => Some(CdcOperation::Insert),
            "u" | "update" | "UPDATE" => Some(CdcOperation::Update),
            "d" | "delete" | "DELETE" => Some(CdcOperation::Delete),
            "r" | "read" | "READ" => Some(CdcOperation::Read),
            "s" | "schema" | "SCHEMA" => Some(CdcOperation::SchemaChange),
            _ => None,
        }
    }
}

/// A single CDC event with before/after row images.
#[derive(Debug, Clone)]
pub struct CdcEvent {
    /// Log Sequence Number — monotonically increasing.
    pub lsn: u64,
    /// Wall-clock timestamp (Unix millis) when the change occurred in the source.
    pub source_ts_ms: i64,
    /// Database/schema/table this event belongs to.
    pub table: String,
    /// The type of change.
    pub operation: CdcOperation,
    /// Row image before the change (None for INSERT/READ).
    pub before: Option<RowImage>,
    /// Row image after the change (None for DELETE).
    pub after: Option<RowImage>,
    /// Optional transaction identifier.
    pub transaction_id: Option<String>,
}

impl CdcEvent {
    /// Create an INSERT event.
    pub fn insert(lsn: u64, ts_ms: i64, table: impl Into<String>, after: RowImage) -> Self {
        CdcEvent {
            lsn,
            source_ts_ms: ts_ms,
            table: table.into(),
            operation: CdcOperation::Insert,
            before: None,
            after: Some(after),
            transaction_id: None,
        }
    }

    /// Create an UPDATE event.
    pub fn update(
        lsn: u64,
        ts_ms: i64,
        table: impl Into<String>,
        before: RowImage,
        after: RowImage,
    ) -> Self {
        CdcEvent {
            lsn,
            source_ts_ms: ts_ms,
            table: table.into(),
            operation: CdcOperation::Update,
            before: Some(before),
            after: Some(after),
            transaction_id: None,
        }
    }

    /// Create a DELETE event.
    pub fn delete(lsn: u64, ts_ms: i64, table: impl Into<String>, before: RowImage) -> Self {
        CdcEvent {
            lsn,
            source_ts_ms: ts_ms,
            table: table.into(),
            operation: CdcOperation::Delete,
            before: Some(before),
            after: None,
            transaction_id: None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CDCLog
// ──────────────────────────────────────────────────────────────────────────────

/// An ordered log of CDC events.
///
/// Events are kept in LSN order. The log supports appending, serialisation
/// to a newline-delimited JSON file, and replay.
#[derive(Debug, Default)]
pub struct CdcLog {
    events: Vec<CdcEvent>,
    /// Next LSN to assign when appending.
    next_lsn: u64,
}

impl CdcLog {
    /// Create an empty log starting at LSN 0.
    pub fn new() -> Self {
        CdcLog {
            events: Vec::new(),
            next_lsn: 0,
        }
    }

    /// Create a log with a specific starting LSN.
    pub fn with_start_lsn(start: u64) -> Self {
        CdcLog {
            events: Vec::new(),
            next_lsn: start,
        }
    }

    /// Append a pre-built event (its existing LSN is preserved; `next_lsn` is
    /// advanced if needed to stay consistent).
    pub fn push(&mut self, event: CdcEvent) {
        if event.lsn >= self.next_lsn {
            self.next_lsn = event.lsn + 1;
        }
        self.events.push(event);
    }

    /// Append an INSERT and auto-assign the next LSN.
    pub fn insert(&mut self, ts_ms: i64, table: impl Into<String>, after: RowImage) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        self.events.push(CdcEvent::insert(lsn, ts_ms, table, after));
        lsn
    }

    /// Append an UPDATE and auto-assign the next LSN.
    pub fn update(
        &mut self,
        ts_ms: i64,
        table: impl Into<String>,
        before: RowImage,
        after: RowImage,
    ) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        self.events
            .push(CdcEvent::update(lsn, ts_ms, table, before, after));
        lsn
    }

    /// Append a DELETE and auto-assign the next LSN.
    pub fn delete(&mut self, ts_ms: i64, table: impl Into<String>, before: RowImage) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        self.events
            .push(CdcEvent::delete(lsn, ts_ms, table, before));
        lsn
    }

    /// Number of events in the log.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns true if the log contains no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Iterate over all events in LSN order.
    pub fn events(&self) -> &[CdcEvent] {
        &self.events
    }

    /// Filter events for a single table.
    pub fn events_for_table(&self, table: &str) -> Vec<&CdcEvent> {
        self.events.iter().filter(|e| e.table == table).collect()
    }

    /// Serialize the log to newline-delimited JSON (NDJSON).
    pub fn write_ndjson<W: Write>(&self, mut writer: W) -> Result<()> {
        for ev in &self.events {
            let line = event_to_ndjson(ev);
            writer
                .write_all(line.as_bytes())
                .map_err(|e| IoError::Io(e))?;
            writer.write_all(b"\n").map_err(|e| IoError::Io(e))?;
        }
        Ok(())
    }

    /// Deserialize a log from newline-delimited JSON produced by `write_ndjson`.
    pub fn read_ndjson<R: BufRead>(reader: R) -> Result<Self> {
        let mut log = CdcLog::new();
        for (line_no, line_res) in reader.lines().enumerate() {
            let line = line_res.map_err(|e| IoError::Io(e))?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let ev = ndjson_to_event(trimmed).map_err(|e| {
                IoError::ParseError(format!("CDC NDJSON line {}: {}", line_no + 1, e))
            })?;
            log.push(ev);
        }
        Ok(log)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Replay
// ──────────────────────────────────────────────────────────────────────────────

/// Replay a `CdcLog` to reconstruct the final state of a table.
///
/// The primary key column(s) identify rows. Returns a map of
/// `pk_value → current_row`.
///
/// * `pk_columns` — column name(s) used as primary key (in order). The key
///   string in the returned map is the PK columns joined with `\x00`.
pub fn replay_cdc(
    log: &CdcLog,
    table: &str,
    pk_columns: &[&str],
) -> Result<HashMap<String, RowImage>> {
    if pk_columns.is_empty() {
        return Err(IoError::ValidationError(
            "pk_columns must not be empty".to_string(),
        ));
    }

    let mut state: HashMap<String, RowImage> = HashMap::new();

    for ev in log.events_for_table(table) {
        match ev.operation {
            CdcOperation::Insert | CdcOperation::Read => {
                if let Some(after) = &ev.after {
                    let pk = extract_pk(after, pk_columns)?;
                    state.insert(pk, after.clone());
                }
            }
            CdcOperation::Update => {
                if let Some(after) = &ev.after {
                    let pk = extract_pk(after, pk_columns)?;
                    state.insert(pk, after.clone());
                }
            }
            CdcOperation::Delete => {
                if let Some(before) = &ev.before {
                    let pk = extract_pk(before, pk_columns)?;
                    state.remove(&pk);
                }
            }
            CdcOperation::SchemaChange => {
                // schema changes don't alter row state
            }
        }
    }

    Ok(state)
}

fn extract_pk(row: &RowImage, pk_columns: &[&str]) -> Result<String> {
    let mut parts = Vec::with_capacity(pk_columns.len());
    for col in pk_columns {
        let val = row.get(*col).ok_or_else(|| {
            IoError::ValidationError(format!("PK column '{}' not found in row image", col))
        })?;
        parts.push(val.to_display());
    }
    Ok(parts.join("\x00"))
}

// ──────────────────────────────────────────────────────────────────────────────
// CDC → columnar materialisation
// ──────────────────────────────────────────────────────────────────────────────

/// Materialised columnar snapshot produced by `cdc_to_parquet`.
#[derive(Debug, Default)]
pub struct CdcColumnarSnapshot {
    /// Column names in stable order.
    pub columns: Vec<String>,
    /// Column data: each inner Vec has one entry per row.
    pub data: HashMap<String, Vec<CdcValue>>,
    /// Number of rows.
    pub row_count: usize,
}

impl CdcColumnarSnapshot {
    fn new(columns: Vec<String>) -> Self {
        let mut data = HashMap::new();
        for col in &columns {
            data.insert(col.clone(), Vec::new());
        }
        CdcColumnarSnapshot {
            columns,
            data,
            row_count: 0,
        }
    }

    fn push_row(&mut self, row: &RowImage) {
        for col in &self.columns {
            let val = row.get(col).cloned().unwrap_or(CdcValue::Null);
            if let Some(col_data) = self.data.get_mut(col) {
                col_data.push(val);
            }
        }
        self.row_count += 1;
    }
}

/// Materialise a CDC log for a given table into a columnar snapshot.
///
/// Equivalent to calling `replay_cdc` and then laying the result out column-
/// wise. `column_order` determines the column ordering; pass `None` to infer
/// from the first row encountered.
pub fn cdc_to_parquet(
    log: &CdcLog,
    table: &str,
    pk_columns: &[&str],
    column_order: Option<&[&str]>,
) -> Result<CdcColumnarSnapshot> {
    let state = replay_cdc(log, table, pk_columns)?;

    // Determine column names
    let columns: Vec<String> = if let Some(order) = column_order {
        order.iter().map(|s| s.to_string()).collect()
    } else {
        // Collect from first row, then sort for determinism
        if let Some(row) = state.values().next() {
            let mut cols: Vec<String> = row.keys().cloned().collect();
            cols.sort();
            cols
        } else {
            return Ok(CdcColumnarSnapshot::default());
        }
    };

    let mut snapshot = CdcColumnarSnapshot::new(columns);

    // Sort rows by PK for deterministic output
    let mut sorted_rows: Vec<(&String, &RowImage)> = state.iter().collect();
    sorted_rows.sort_by_key(|(pk, _)| pk.as_str());

    for (_, row) in sorted_rows {
        snapshot.push_row(row);
    }

    Ok(snapshot)
}

// ──────────────────────────────────────────────────────────────────────────────
// Debezium JSON parser
// ──────────────────────────────────────────────────────────────────────────────

/// Parse a Debezium-format CDC JSON string into a `CdcEvent`.
///
/// Supports the standard Debezium envelope:
/// ```json
/// {
///   "payload": {
///     "op": "c",
///     "ts_ms": 1234567890000,
///     "source": { "table": "orders", ... },
///     "before": null,
///     "after": { "id": 1, "name": "Alice" }
///   }
/// }
/// ```
pub fn debezium_json_parser(json: &str, lsn: u64) -> Result<CdcEvent> {
    let trimmed = json.trim();
    // Walk into "payload" if present; otherwise treat the root as the payload.
    let payload_str = extract_payload_str(trimmed);

    let op_str = extract_string_field(payload_str, "op")
        .ok_or_else(|| IoError::ParseError("Debezium JSON missing 'op' field".to_string()))?;

    let operation = CdcOperation::from_str(&op_str)
        .ok_or_else(|| IoError::ParseError(format!("Unknown Debezium op '{}'", op_str)))?;

    let ts_ms = extract_i64_field(payload_str, "ts_ms").unwrap_or(0);

    // Extract table from source.table
    let table = extract_source_table(payload_str).unwrap_or_else(|| "unknown".to_string());

    let before = extract_row_object(payload_str, "before");
    let after = extract_row_object(payload_str, "after");

    Ok(CdcEvent {
        lsn,
        source_ts_ms: ts_ms,
        table,
        operation,
        before,
        after,
        transaction_id: extract_string_field(payload_str, "transaction"),
    })
}

/// Parse a CDC log file where each line is a Debezium JSON event.
pub fn parse_debezium_log<P: AsRef<Path>>(path: P) -> Result<CdcLog> {
    let file = std::fs::File::open(path.as_ref()).map_err(|e| IoError::Io(e))?;
    let reader = BufReader::new(file);
    let mut log = CdcLog::new();
    let mut lsn: u64 = 0;

    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|e| IoError::Io(e))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let ev = debezium_json_parser(trimmed, lsn)
            .map_err(|e| IoError::ParseError(format!("Debezium line {}: {}", line_no + 1, e)))?;
        lsn = ev.lsn + 1;
        log.push(ev);
    }
    Ok(log)
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal serialisation helpers
// ──────────────────────────────────────────────────────────────────────────────

fn cdc_value_to_json(v: &CdcValue) -> String {
    match v {
        CdcValue::Null => "null".to_string(),
        CdcValue::Bool(b) => b.to_string(),
        CdcValue::Int(i) => i.to_string(),
        CdcValue::Float(f) => {
            if f.is_nan() || f.is_infinite() {
                "null".to_string()
            } else {
                f.to_string()
            }
        }
        CdcValue::Text(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        CdcValue::Bytes(b) => {
            // Base-64-like hex encoding for simplicity
            let hex: String = b.iter().map(|byte| format!("{:02x}", byte)).collect();
            format!("\"\\u0000hex:{hex}\"")
        }
    }
}

fn row_image_to_json(row: &RowImage) -> String {
    let mut pairs: Vec<String> = row
        .iter()
        .map(|(k, v)| format!("\"{}\":{}", k, cdc_value_to_json(v)))
        .collect();
    pairs.sort(); // deterministic order
    format!("{{{}}}", pairs.join(","))
}

fn event_to_ndjson(ev: &CdcEvent) -> String {
    let op = match ev.operation {
        CdcOperation::Insert => "c",
        CdcOperation::Update => "u",
        CdcOperation::Delete => "d",
        CdcOperation::Read => "r",
        CdcOperation::SchemaChange => "s",
    };
    let before_str = ev
        .before
        .as_ref()
        .map(row_image_to_json)
        .unwrap_or_else(|| "null".to_string());
    let after_str = ev
        .after
        .as_ref()
        .map(row_image_to_json)
        .unwrap_or_else(|| "null".to_string());
    let tx = ev
        .transaction_id
        .as_deref()
        .map(|t| format!("\"{}\"", t))
        .unwrap_or_else(|| "null".to_string());

    format!(
        "{{\"lsn\":{},\"ts_ms\":{},\"table\":\"{}\",\"op\":\"{}\",\"before\":{},\"after\":{},\"tx\":{}}}",
        ev.lsn, ev.source_ts_ms, ev.table, op, before_str, after_str, tx
    )
}

fn ndjson_to_event(s: &str) -> std::result::Result<CdcEvent, String> {
    let lsn = extract_u64_field(s, "lsn").ok_or_else(|| "missing 'lsn'".to_string())?;
    let ts_ms = extract_i64_field(s, "ts_ms").unwrap_or(0);
    let table = extract_string_field(s, "table").ok_or_else(|| "missing 'table'".to_string())?;
    let op_str = extract_string_field(s, "op").ok_or_else(|| "missing 'op'".to_string())?;
    let operation =
        CdcOperation::from_str(&op_str).ok_or_else(|| format!("unknown op '{}'", op_str))?;
    let before = extract_row_object(s, "before");
    let after = extract_row_object(s, "after");
    let transaction_id = extract_string_field(s, "tx");

    Ok(CdcEvent {
        lsn,
        source_ts_ms: ts_ms,
        table,
        operation,
        before,
        after,
        transaction_id,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Minimal JSON extraction helpers (no external JSON crate dependency)
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the content of a `"payload": {...}` sub-object, or the whole input.
fn extract_payload_str(s: &str) -> &str {
    if let Some(start) = find_key_value_start(s, "payload") {
        // find the matching `{`
        if let Some(obj_start) = s[start..].find('{') {
            let abs = start + obj_start;
            if let Some(end) = matching_brace(s, abs) {
                return &s[abs..=end];
            }
        }
    }
    s
}

fn extract_string_field(s: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let pos = s.find(&pattern)?;
    let after_key = &s[pos + pattern.len()..];
    let colon_pos = after_key.find(':')? + 1;
    let val_str = after_key[colon_pos..].trim_start();
    if let Some(inner) = val_str.strip_prefix('"') {
        // quoted string
        let mut result = String::new();
        let mut chars = inner.chars();
        loop {
            match chars.next() {
                None => break,
                Some('"') => break,
                Some('\\') => match chars.next() {
                    Some('"') => result.push('"'),
                    Some('\\') => result.push('\\'),
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some(c) => {
                        result.push('\\');
                        result.push(c);
                    }
                    None => break,
                },
                Some(c) => result.push(c),
            }
        }
        Some(result)
    } else {
        None
    }
}

fn extract_i64_field(s: &str, key: &str) -> Option<i64> {
    let pattern = format!("\"{}\"", key);
    let pos = s.find(&pattern)?;
    let after_key = &s[pos + pattern.len()..];
    let colon_pos = after_key.find(':')? + 1;
    let val_str = after_key[colon_pos..].trim_start();
    // Read digits (possibly with leading minus)
    let end = val_str
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(val_str.len());
    val_str[..end].parse::<i64>().ok()
}

fn extract_u64_field(s: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{}\"", key);
    let pos = s.find(&pattern)?;
    let after_key = &s[pos + pattern.len()..];
    let colon_pos = after_key.find(':')? + 1;
    let val_str = after_key[colon_pos..].trim_start();
    let end = val_str
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(val_str.len());
    val_str[..end].parse::<u64>().ok()
}

fn extract_source_table(s: &str) -> Option<String> {
    // Try "source": { ... "table": "x" ... }
    let source_pos = find_key_value_start(s, "source")?;
    let source_str = &s[source_pos..];
    let obj_start = source_str.find('{')?;
    let abs = source_pos + obj_start;
    let end = matching_brace(s, abs)?;
    let source_obj = &s[abs..=end];
    extract_string_field(source_obj, "table")
}

fn extract_row_object(s: &str, key: &str) -> Option<RowImage> {
    let pattern = format!("\"{}\"", key);
    let pos = s.find(&pattern)?;
    let after_key = &s[pos + pattern.len()..];
    let colon_pos = after_key.find(':')? + 1;
    let val_str = after_key[colon_pos..].trim_start();

    if val_str.starts_with("null") || val_str.is_empty() {
        return None;
    }

    if !val_str.starts_with('{') {
        return None;
    }

    // The absolute position within `s` where the object starts
    let abs_start =
        s.len() - after_key.len() + colon_pos + (after_key[colon_pos..].len() - val_str.len());
    let abs_end = matching_brace(s, abs_start)?;
    let obj_str = &s[abs_start..=abs_end];

    parse_flat_json_object(obj_str)
}

/// Parse a flat (single-level) JSON object into a RowImage.
fn parse_flat_json_object(s: &str) -> Option<RowImage> {
    let inner = s
        .strip_prefix('{')
        .and_then(|t| t.strip_suffix('}'))
        .unwrap_or(s);

    let mut map = HashMap::new();
    let mut rest = inner.trim();

    while !rest.is_empty() {
        // Read key
        rest = rest.trim_start_matches(',').trim();
        if rest.is_empty() {
            break;
        }
        if !rest.starts_with('"') {
            break;
        }
        let (key, after_key) = parse_json_string(rest)?;
        rest = after_key.trim_start();
        if !rest.starts_with(':') {
            break;
        }
        rest = rest[1..].trim_start();

        // Read value
        let (val, after_val) = parse_json_value(rest)?;
        map.insert(key, val);
        rest = after_val.trim_start();
    }

    Some(map)
}

fn parse_json_string(s: &str) -> Option<(String, &str)> {
    if !s.starts_with('"') {
        return None;
    }
    let mut result = String::new();
    let mut chars = s[1..].char_indices();
    loop {
        match chars.next() {
            None => return None,
            Some((i, '"')) => {
                // +2: leading `"` + closing `"`
                return Some((result, &s[i + 2..]));
            }
            Some((_, '\\')) => match chars.next() {
                Some((_, '"')) => result.push('"'),
                Some((_, '\\')) => result.push('\\'),
                Some((_, 'n')) => result.push('\n'),
                Some((_, 'r')) => result.push('\r'),
                Some((_, 't')) => result.push('\t'),
                Some((_, c)) => {
                    result.push('\\');
                    result.push(c);
                }
                None => return None,
            },
            Some((_, c)) => result.push(c),
        }
    }
}

fn parse_json_value(s: &str) -> Option<(CdcValue, &str)> {
    if let Some(rest) = s.strip_prefix("null") {
        return Some((CdcValue::Null, rest));
    }
    if let Some(rest) = s.strip_prefix("true") {
        return Some((CdcValue::Bool(true), rest));
    }
    if let Some(rest) = s.strip_prefix("false") {
        return Some((CdcValue::Bool(false), rest));
    }
    if s.starts_with('"') {
        let (string, rest) = parse_json_string(s)?;
        return Some((CdcValue::Text(string), rest));
    }
    // Number: integer or float
    let end = s
        .find(|c: char| {
            !c.is_ascii_digit() && c != '-' && c != '.' && c != 'e' && c != 'E' && c != '+'
        })
        .unwrap_or(s.len());
    let num_str = &s[..end];
    if num_str.is_empty() {
        return None;
    }
    if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
        let f = num_str.parse::<f64>().ok()?;
        Some((CdcValue::Float(f), &s[end..]))
    } else {
        let i = num_str.parse::<i64>().ok()?;
        Some((CdcValue::Int(i), &s[end..]))
    }
}

/// Find the position in `s` where the value for key `key` starts.
fn find_key_value_start(s: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{}\"", key);
    let pos = s.find(&pattern)?;
    Some(pos)
}

/// Return the index of the `}` that matches the `{` at `start` in `s`.
fn matching_brace(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.get(start) != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escape = false;
    for (i, &b) in bytes[start..].iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if b == b'\\' && in_str {
            escape = true;
            continue;
        }
        if b == b'"' {
            in_str = !in_str;
            continue;
        }
        if in_str {
            continue;
        }
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(start + i);
                }
            }
            _ => {}
        }
    }
    None
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(pairs: &[(&str, CdcValue)]) -> RowImage {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn test_cdc_log_basic() {
        let mut log = CdcLog::new();
        let after = make_row(&[
            ("id", CdcValue::Int(1)),
            ("name", CdcValue::Text("Alice".into())),
        ]);
        let lsn = log.insert(1_000, "users", after);
        assert_eq!(lsn, 0);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_replay_insert_delete() {
        let mut log = CdcLog::new();

        let row1 = make_row(&[
            ("id", CdcValue::Int(1)),
            ("val", CdcValue::Text("a".into())),
        ]);
        let row2 = make_row(&[
            ("id", CdcValue::Int(2)),
            ("val", CdcValue::Text("b".into())),
        ]);

        log.insert(0, "t", row1.clone());
        log.insert(0, "t", row2.clone());
        log.delete(0, "t", row1.clone());

        let state = replay_cdc(&log, "t", &["id"]).expect("replay failed");
        assert_eq!(state.len(), 1);
        assert!(state.contains_key("2"));
    }

    #[test]
    fn test_replay_update() {
        let mut log = CdcLog::new();

        let before = make_row(&[
            ("id", CdcValue::Int(1)),
            ("val", CdcValue::Text("old".into())),
        ]);
        let after = make_row(&[
            ("id", CdcValue::Int(1)),
            ("val", CdcValue::Text("new".into())),
        ]);

        log.insert(0, "t", before.clone());
        log.update(0, "t", before, after);

        let state = replay_cdc(&log, "t", &["id"]).expect("replay failed");
        let row = state.get("1").expect("row must exist");
        assert_eq!(row.get("val"), Some(&CdcValue::Text("new".into())));
    }

    #[test]
    fn test_ndjson_roundtrip() {
        let mut log = CdcLog::new();
        let row = make_row(&[
            ("id", CdcValue::Int(42)),
            ("x", CdcValue::Float(std::f64::consts::PI)),
        ]);
        log.insert(9999, "metrics", row);

        let mut buf = Vec::new();
        log.write_ndjson(&mut buf).expect("write failed");

        let log2 = CdcLog::read_ndjson(std::io::BufReader::new(&buf[..])).expect("read failed");
        assert_eq!(log2.len(), 1);
        let ev = &log2.events()[0];
        assert_eq!(ev.operation, CdcOperation::Insert);
        assert_eq!(ev.table, "metrics");
    }

    #[test]
    fn test_debezium_parser() {
        let json = r#"{
            "payload": {
                "op": "c",
                "ts_ms": 1711234567890,
                "source": {"table": "orders", "db": "mydb"},
                "before": null,
                "after": {"id": 99, "amount": 10}
            }
        }"#;
        let ev = debezium_json_parser(json, 7).expect("parse failed");
        assert_eq!(ev.operation, CdcOperation::Insert);
        assert_eq!(ev.table, "orders");
        assert_eq!(ev.lsn, 7);
        assert!(ev.before.is_none());
        assert!(ev.after.is_some());
    }

    #[test]
    fn test_cdc_to_parquet() {
        let mut log = CdcLog::new();
        for i in 0..5i64 {
            let row = make_row(&[
                ("id", CdcValue::Int(i)),
                ("v", CdcValue::Float(i as f64 * 1.5)),
            ]);
            log.insert(0, "data", row);
        }

        let snap =
            cdc_to_parquet(&log, "data", &["id"], Some(&["id", "v"])).expect("materialise failed");
        assert_eq!(snap.row_count, 5);
        assert_eq!(snap.columns.len(), 2);
    }
}
