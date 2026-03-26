//! Lance columnar format binary writer and reader.
//!
//! Binary layout
//! ─────────────
//! File = Magic(8) + SchemaLen(4 LE u32) + SchemaJSON(N) + Batch* + Footer
//! Batch = NumRows(4 LE u32) + Column*
//! Column = TypeTag(1) + Nullable(1: 0 or 1) + DataLen(4 LE u32) + Data(N)
//!          [if Nullable == 1: ValidityLen(4) + ValidityBytes follows Data]
//! Footer = BatchCount(4 LE u32) + FooterMagic(8)
//!
//! String encoding inside a column's `Data` block:
//!   for each string: Len(4 LE u32) + UTF-8 bytes
//! Boolean: 1 byte per element (0 = false, 1 = true)
//! All other primitives: little-endian raw bytes.

use std::io::{self, Read, Seek, SeekFrom, Write};

use serde_json;

use super::types::{LanceBatch, LanceColumn, LanceDataType, LanceSchema};

/// Magic bytes at the start of every Lance file.
pub const LANCE_MAGIC: &[u8; 8] = b"LANCE001";
/// Footer magic bytes at the end of every Lance file.
pub const LANCE_EOF: &[u8; 8] = b"LANCEEOF";

// ─────────────────────────────────────────────────────────────────────────────
// LanceWriter
// ─────────────────────────────────────────────────────────────────────────────

/// Writes Lance batches to any `Write` sink.
pub struct LanceWriter<W: Write> {
    writer: W,
    schema: LanceSchema,
    batches_written: usize,
}

impl<W: Write> LanceWriter<W> {
    /// Create a new writer, emitting the file header (magic + schema).
    pub fn new(mut writer: W, schema: LanceSchema) -> io::Result<Self> {
        // Magic
        writer.write_all(LANCE_MAGIC)?;
        // Schema JSON with 4-byte length prefix
        let schema_json = serde_json::to_vec(&schema)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let schema_len = schema_json.len() as u32;
        writer.write_all(&schema_len.to_le_bytes())?;
        writer.write_all(&schema_json)?;
        Ok(Self {
            writer,
            schema,
            batches_written: 0,
        })
    }

    /// Write a single batch.
    pub fn write_batch(&mut self, batch: &LanceBatch) -> io::Result<()> {
        let num_rows = batch.num_rows as u32;
        self.writer.write_all(&num_rows.to_le_bytes())?;
        for col in &batch.columns {
            write_column(&mut self.writer, col)?;
        }
        self.batches_written += 1;
        Ok(())
    }

    /// Finalise the file by writing the footer and returning the inner writer.
    pub fn finish(mut self) -> io::Result<W> {
        let batch_count = self.batches_written as u32;
        self.writer.write_all(&batch_count.to_le_bytes())?;
        self.writer.write_all(LANCE_EOF)?;
        Ok(self.writer)
    }

    /// Reference to the schema.
    pub fn schema(&self) -> &LanceSchema {
        &self.schema
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LanceReader
// ─────────────────────────────────────────────────────────────────────────────

/// Reads Lance batches from any `Read + Seek` source.
pub struct LanceReader<R: Read + Seek> {
    reader: R,
    schema: LanceSchema,
    /// Offset just after the schema block (= start of first batch).
    data_start: u64,
    /// Total number of batches in the file (read from footer).
    total_batches: usize,
    /// How many batches have been read by sequential `read_batch` calls.
    batches_read: usize,
}

impl<R: Read + Seek> LanceReader<R> {
    /// Open a Lance reader, validating the magic and reading the schema.
    ///
    /// Seeks to the footer to determine the total number of batches, then
    /// rewinds to the first batch so that `read_batch` works correctly.
    pub fn new(mut reader: R) -> io::Result<Self> {
        // Check magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != LANCE_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Not a Lance file: bad magic bytes",
            ));
        }
        // Schema
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let schema_len = u32::from_le_bytes(len_buf) as usize;
        let mut schema_bytes = vec![0u8; schema_len];
        reader.read_exact(&mut schema_bytes)?;
        let schema: LanceSchema = serde_json::from_slice(&schema_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let data_start = reader.stream_position()?;

        // Read the footer to learn total_batches.
        // Footer layout: BatchCount(4 LE u32) + LANCEEOF(8) = 12 bytes from the end.
        let total_batches = match reader.seek(SeekFrom::End(-12)) {
            Ok(_) => {
                let mut fbuf = [0u8; 4];
                reader.read_exact(&mut fbuf)?;
                let mut eofbuf = [0u8; 8];
                reader.read_exact(&mut eofbuf)?;
                if &eofbuf == LANCE_EOF {
                    u32::from_le_bytes(fbuf) as usize
                } else {
                    0
                }
            }
            Err(_) => 0,
        };

        // Rewind to the first batch.
        reader.seek(SeekFrom::Start(data_start))?;

        Ok(Self {
            reader,
            schema,
            data_start,
            total_batches,
            batches_read: 0,
        })
    }

    /// The file schema.
    pub fn schema(&self) -> &LanceSchema {
        &self.schema
    }

    /// Total number of batches in the file.
    pub fn total_batches(&self) -> usize {
        self.total_batches
    }

    /// Read the next batch sequentially.
    ///
    /// Returns `Ok(None)` after all `total_batches` batches have been read.
    pub fn read_batch(&mut self) -> io::Result<Option<LanceBatch>> {
        if self.batches_read >= self.total_batches {
            return Ok(None);
        }

        let mut buf4 = [0u8; 4];
        match self.reader.read_exact(&mut buf4) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        let num_rows = u32::from_le_bytes(buf4) as usize;

        let mut columns = Vec::with_capacity(self.schema.fields.len());
        for _field in &self.schema.fields {
            match read_column(&mut self.reader, num_rows) {
                Ok(col) => columns.push(col),
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(e) => return Err(e),
            }
        }

        self.batches_read += 1;
        Ok(Some(LanceBatch::new(
            self.schema.clone(),
            columns,
            num_rows,
        )))
    }

    /// Read all batches into a `Vec`, rewinding to the beginning first.
    pub fn collect_all(&mut self) -> io::Result<Vec<LanceBatch>> {
        self.reader.seek(SeekFrom::Start(self.data_start))?;
        self.batches_read = 0;
        let mut batches = Vec::new();
        while let Some(b) = self.read_batch()? {
            batches.push(b);
        }
        Ok(batches)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Column serialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

fn write_column<W: Write>(w: &mut W, col: &LanceColumn) -> io::Result<()> {
    match col {
        LanceColumn::Nullable(inner, validity) => {
            // type_tag of inner, nullable flag = 1
            w.write_all(&[inner.data_type().type_tag(), 1u8])?;
            let raw = encode_column_data(inner)?;
            w.write_all(&(raw.len() as u32).to_le_bytes())?;
            w.write_all(&raw)?;
            // validity bitmap: 1 byte per element
            let validity_bytes: Vec<u8> = validity.iter().map(|&b| b as u8).collect();
            w.write_all(&(validity_bytes.len() as u32).to_le_bytes())?;
            w.write_all(&validity_bytes)?;
        }
        _ => {
            w.write_all(&[col.data_type().type_tag(), 0u8])?;
            let raw = encode_column_data(col)?;
            w.write_all(&(raw.len() as u32).to_le_bytes())?;
            w.write_all(&raw)?;
        }
    }
    Ok(())
}

fn encode_column_data(col: &LanceColumn) -> io::Result<Vec<u8>> {
    match col {
        LanceColumn::Float32(v) => {
            let mut buf = Vec::with_capacity(v.len() * 4);
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            Ok(buf)
        }
        LanceColumn::Float64(v) => {
            let mut buf = Vec::with_capacity(v.len() * 8);
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            Ok(buf)
        }
        LanceColumn::Int32(v) => {
            let mut buf = Vec::with_capacity(v.len() * 4);
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            Ok(buf)
        }
        LanceColumn::Int64(v) => {
            let mut buf = Vec::with_capacity(v.len() * 8);
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            Ok(buf)
        }
        LanceColumn::UInt32(v) => {
            let mut buf = Vec::with_capacity(v.len() * 4);
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            Ok(buf)
        }
        LanceColumn::UInt64(v) => {
            let mut buf = Vec::with_capacity(v.len() * 8);
            for &x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            Ok(buf)
        }
        LanceColumn::Utf8(v) => {
            let mut buf = Vec::new();
            for s in v {
                let bytes = s.as_bytes();
                buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(bytes);
            }
            Ok(buf)
        }
        LanceColumn::Boolean(v) => Ok(v.iter().map(|&b| b as u8).collect()),
        LanceColumn::Nullable(inner, _) => encode_column_data(inner),
    }
}

fn read_column<R: Read>(r: &mut R, num_rows: usize) -> io::Result<LanceColumn> {
    let mut hdr = [0u8; 2];
    r.read_exact(&mut hdr)?;
    let type_tag = hdr[0];
    let nullable_flag = hdr[1] != 0;

    let dtype = LanceDataType::from_type_tag(type_tag).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown Lance type tag: {type_tag}"),
        )
    })?;

    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let data_len = u32::from_le_bytes(len_buf) as usize;
    let mut raw = vec![0u8; data_len];
    r.read_exact(&mut raw)?;

    let col = decode_column_data(&raw, &dtype, num_rows)?;

    if nullable_flag {
        let mut vlen_buf = [0u8; 4];
        r.read_exact(&mut vlen_buf)?;
        let vlen = u32::from_le_bytes(vlen_buf) as usize;
        let mut validity_raw = vec![0u8; vlen];
        r.read_exact(&mut validity_raw)?;
        let validity: Vec<bool> = validity_raw.iter().map(|&b| b != 0).collect();
        Ok(LanceColumn::Nullable(Box::new(col), validity))
    } else {
        Ok(col)
    }
}

fn decode_column_data(
    raw: &[u8],
    dtype: &LanceDataType,
    num_rows: usize,
) -> io::Result<LanceColumn> {
    match dtype {
        LanceDataType::Float32 => {
            if raw.len() < num_rows * 4 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Float32 data too short",
                ));
            }
            let v: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
                .collect();
            Ok(LanceColumn::Float32(v))
        }
        LanceDataType::Float64 => {
            if raw.len() < num_rows * 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Float64 data too short",
                ));
            }
            let v: Vec<f64> = raw
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
                .collect();
            Ok(LanceColumn::Float64(v))
        }
        LanceDataType::Int32 => {
            let v: Vec<i32> = raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
                .collect();
            Ok(LanceColumn::Int32(v))
        }
        LanceDataType::Int64 => {
            let v: Vec<i64> = raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
                .collect();
            Ok(LanceColumn::Int64(v))
        }
        LanceDataType::UInt32 => {
            let v: Vec<u32> = raw
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
                .collect();
            Ok(LanceColumn::UInt32(v))
        }
        LanceDataType::UInt64 => {
            let v: Vec<u64> = raw
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
                .collect();
            Ok(LanceColumn::UInt64(v))
        }
        LanceDataType::Utf8 => {
            let mut strings = Vec::with_capacity(num_rows);
            let mut pos = 0usize;
            while pos + 4 <= raw.len() {
                let slen =
                    u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap_or([0; 4])) as usize;
                pos += 4;
                if pos + slen > raw.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Utf8 string out of bounds",
                    ));
                }
                let s = String::from_utf8(raw[pos..pos + slen].to_vec())
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                strings.push(s);
                pos += slen;
            }
            Ok(LanceColumn::Utf8(strings))
        }
        LanceDataType::Boolean => {
            let v: Vec<bool> = raw.iter().map(|&b| b != 0).collect();
            Ok(LanceColumn::Boolean(v))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lance::types::{LanceField, LanceSchema};
    use std::io::Cursor;

    fn simple_schema() -> LanceSchema {
        LanceSchema::new(vec![
            LanceField::new("x", LanceDataType::Float64),
            LanceField::new("label", LanceDataType::Utf8),
        ])
    }

    #[test]
    fn test_lance_magic_bytes_present() {
        let schema = simple_schema();
        let mut buf = Vec::new();
        let writer = LanceWriter::new(&mut buf, schema).expect("create writer");
        writer.finish().expect("finish");
        assert_eq!(&buf[..8], LANCE_MAGIC);
    }

    #[test]
    fn test_lance_write_read_float64_column() {
        let schema = LanceSchema::new(vec![LanceField::new("v", LanceDataType::Float64)]);
        let mut buf = Vec::new();
        {
            let mut writer = LanceWriter::new(&mut buf, schema.clone()).expect("create writer");
            let batch = LanceBatch::new(
                schema.clone(),
                vec![LanceColumn::Float64(vec![1.0, 2.0, 3.0])],
                3,
            );
            writer.write_batch(&batch).expect("write");
            writer.finish().expect("finish");
        }
        let cursor = Cursor::new(buf);
        let mut reader = LanceReader::new(cursor).expect("open reader");
        let batches = reader.collect_all().expect("collect");
        assert_eq!(batches.len(), 1);
        match &batches[0].columns[0] {
            LanceColumn::Float64(v) => assert_eq!(v, &[1.0, 2.0, 3.0]),
            _ => panic!("unexpected column type"),
        }
    }

    #[test]
    fn test_lance_write_read_utf8_column() {
        let schema = LanceSchema::new(vec![LanceField::new("name", LanceDataType::Utf8)]);
        let mut buf = Vec::new();
        {
            let mut writer = LanceWriter::new(&mut buf, schema.clone()).expect("create");
            let batch = LanceBatch::new(
                schema.clone(),
                vec![LanceColumn::Utf8(vec![
                    "hello".into(),
                    "world".into(),
                    "lance".into(),
                ])],
                3,
            );
            writer.write_batch(&batch).expect("write");
            writer.finish().expect("finish");
        }
        let mut reader = LanceReader::new(Cursor::new(buf)).expect("open");
        let batches = reader.collect_all().expect("collect");
        assert_eq!(batches.len(), 1);
        match &batches[0].columns[0] {
            LanceColumn::Utf8(v) => {
                assert_eq!(v[0], "hello");
                assert_eq!(v[2], "lance");
            }
            _ => panic!("unexpected column type"),
        }
    }

    #[test]
    fn test_lance_multiple_batches() {
        let schema = LanceSchema::new(vec![LanceField::new("n", LanceDataType::Int32)]);
        let mut buf = Vec::new();
        {
            let mut writer = LanceWriter::new(&mut buf, schema.clone()).expect("create");
            for i in 0..3u32 {
                let batch =
                    LanceBatch::new(schema.clone(), vec![LanceColumn::Int32(vec![i as i32])], 1);
                writer.write_batch(&batch).expect("write batch");
            }
            writer.finish().expect("finish");
        }
        let mut reader = LanceReader::new(Cursor::new(buf)).expect("open");
        let batches = reader.collect_all().expect("collect");
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_lance_empty_batch() {
        let schema = LanceSchema::new(vec![LanceField::new("x", LanceDataType::Float32)]);
        let mut buf = Vec::new();
        {
            let mut writer = LanceWriter::new(&mut buf, schema.clone()).expect("create");
            let batch = LanceBatch::new(schema.clone(), vec![LanceColumn::Float32(Vec::new())], 0);
            writer.write_batch(&batch).expect("write");
            writer.finish().expect("finish");
        }
        let mut reader = LanceReader::new(Cursor::new(buf)).expect("open");
        let batches = reader.collect_all().expect("collect");
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows, 0);
    }

    #[test]
    fn test_lance_schema_roundtrip() {
        let schema = simple_schema();
        let json = serde_json::to_vec(&schema).expect("serialize");
        let parsed: LanceSchema = serde_json::from_slice(&json).expect("deserialize");
        assert_eq!(parsed.fields.len(), 2);
        assert_eq!(parsed.fields[0].name, "x");
        assert_eq!(parsed.fields[1].name, "label");
    }

    #[test]
    fn test_lance_footer_eof_magic() {
        let schema = LanceSchema::new(vec![]);
        let mut buf = Vec::new();
        {
            let writer = LanceWriter::new(&mut buf, schema).expect("create");
            writer.finish().expect("finish");
        }
        let tail = &buf[buf.len() - 8..];
        assert_eq!(tail, LANCE_EOF);
    }
}
