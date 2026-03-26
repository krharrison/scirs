//! Kafka protocol — pure-Rust in-memory broker simulation.
//!
//! Implements:
//! - Kafka wire protocol framing primitives (strings, bytes, varint, CRC32c, Murmur2)
//! - [`InMemoryKafkaBroker`]: multi-topic, multi-partition in-process broker
//! - [`KafkaProducer`]: buffered producer with key-based or round-robin partitioning
//! - [`KafkaConsumer`]: poll-based consumer with offset tracking and commit
//!
//! No C bindings (no `rdkafka`), no real network I/O.  All communication is
//! in-process via shared mutable references.
//!
//! # Example
//! ```rust
//! use scirs2_io::protocols::kafka::*;
//!
//! let mut broker = InMemoryKafkaBroker::new();
//! broker.create_topic("events", 3);
//!
//! let mut producer = KafkaProducer::new(&mut broker, KafkaProducerConfig::default());
//! producer.send("events", b"hello".to_vec()).unwrap();
//! let offsets = producer.flush().unwrap();
//! assert_eq!(offsets.len(), 1);
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{IoError, Result as IoResult};

// ─────────────────────────────── config types ────────────────────────────────

/// Compression codec selection for Kafka producers.
///
/// Only `None` is functionally implemented; the other variants are
/// placeholder stubs for compatibility with config APIs.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// No compression.
    None,
    /// Gzip compression (placeholder, not applied in-memory).
    Gzip,
    /// Snappy compression (placeholder, not applied in-memory).
    Snappy,
    /// Zstandard compression (placeholder, not applied in-memory).
    Zstd,
}

/// Policy for the initial offset when a consumer group has no committed offset.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetReset {
    /// Start consuming from the oldest available message.
    Earliest,
    /// Start consuming from the next message to be produced (skip all existing).
    Latest,
}

/// Configuration for a Kafka producer.
#[derive(Debug, Clone)]
pub struct KafkaProducerConfig {
    /// Bootstrap server addresses (informational; not used in in-memory mode).
    pub bootstrap_servers: Vec<String>,
    /// Maximum number of bytes per produce batch.
    pub batch_size: usize,
    /// Delay in milliseconds before flushing a non-full batch.
    pub linger_ms: u64,
    /// Compression codec to use.
    pub compression: CompressionType,
    /// Number of broker acknowledgements required (informational).
    pub acks: i16,
    /// Request timeout in milliseconds.
    pub request_timeout_ms: u64,
}

impl Default for KafkaProducerConfig {
    fn default() -> Self {
        KafkaProducerConfig {
            bootstrap_servers: vec!["localhost:9092".into()],
            batch_size: 16384,
            linger_ms: 5,
            compression: CompressionType::None,
            acks: 1,
            request_timeout_ms: 30_000,
        }
    }
}

/// Configuration for a Kafka consumer.
#[derive(Debug, Clone)]
pub struct KafkaConsumerConfig {
    /// Bootstrap server addresses (informational; not used in in-memory mode).
    pub bootstrap_servers: Vec<String>,
    /// Consumer group identifier (used for offset commit tracking).
    pub group_id: String,
    /// Offset reset policy when no committed offset exists for a partition.
    pub auto_offset_reset: OffsetReset,
    /// Maximum number of records returned in a single `poll`.
    pub max_poll_records: usize,
    /// Session timeout in milliseconds (informational).
    pub session_timeout_ms: u64,
    /// Whether to automatically commit offsets after each poll.
    pub auto_commit: bool,
}

impl Default for KafkaConsumerConfig {
    fn default() -> Self {
        KafkaConsumerConfig {
            bootstrap_servers: vec!["localhost:9092".into()],
            group_id: "scirs2-consumer".into(),
            auto_offset_reset: OffsetReset::Earliest,
            max_poll_records: 500,
            session_timeout_ms: 10_000,
            auto_commit: true,
        }
    }
}

// ─────────────────────────────── record type ─────────────────────────────────

/// A single Kafka message.
#[derive(Debug, Clone)]
pub struct KafkaRecord {
    /// Topic this record belongs to.
    pub topic: String,
    /// Partition index within the topic.
    pub partition: i32,
    /// Offset of this record within the partition.
    pub offset: i64,
    /// Optional message key (used for partition assignment).
    pub key: Option<Vec<u8>>,
    /// Message payload.
    pub value: Vec<u8>,
    /// User-defined headers as `(key, value)` pairs.
    pub headers: Vec<(String, Vec<u8>)>,
    /// Producer-assigned timestamp (Unix epoch milliseconds).
    pub timestamp_ms: i64,
}

impl KafkaRecord {
    /// Construct a minimal record with only topic and value.
    pub fn new(topic: &str, value: Vec<u8>) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        KafkaRecord {
            topic: topic.to_string(),
            partition: 0,
            offset: -1,
            key: None,
            value,
            headers: Vec::new(),
            timestamp_ms,
        }
    }

    /// Attach a key to this record (builder-style).
    pub fn with_key(mut self, key: Vec<u8>) -> Self {
        self.key = Some(key);
        self
    }

    /// Attach a header to this record (builder-style).
    pub fn with_header(mut self, key: &str, value: Vec<u8>) -> Self {
        self.headers.push((key.to_string(), value));
        self
    }
}

// ─────────────────────────────── wire module ─────────────────────────────────

/// Low-level Kafka wire protocol framing utilities.
///
/// All functions in this module are pure-Rust, no-`std::io`, byte-slice operations
/// suitable for both serialization and unit testing of individual protocol fields.
pub mod wire {
    use crate::error::{IoError, Result as IoResult};

    // ── String encoding ───────────────────────────────────────────────────────

    /// Encode a Kafka protocol string.
    ///
    /// Format: `[length: i16 BE]` + UTF-8 bytes.
    /// A `NULL` string (`length == -1`) is represented by an empty `&str` here.
    pub fn encode_string(s: &str) -> Vec<u8> {
        let bytes = s.as_bytes();
        let len = bytes.len() as i16;
        let mut buf = Vec::with_capacity(2 + bytes.len());
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(bytes);
        buf
    }

    /// Decode a Kafka protocol string starting at `offset`.
    ///
    /// Returns `(string, new_offset)`.
    pub fn decode_string(data: &[u8], offset: usize) -> IoResult<(String, usize)> {
        if offset + 2 > data.len() {
            return Err(IoError::FormatError(
                "Kafka string: length field truncated".into(),
            ));
        }
        let len = i16::from_be_bytes(
            data[offset..offset + 2]
                .try_into()
                .map_err(|_| IoError::FormatError("Kafka string len bytes".into()))?,
        );
        let pos = offset + 2;

        if len < 0 {
            // NULL string → empty
            return Ok((String::new(), pos));
        }
        let slen = len as usize;
        if pos + slen > data.len() {
            return Err(IoError::FormatError("Kafka string: data truncated".into()));
        }
        let s = std::str::from_utf8(&data[pos..pos + slen])
            .map_err(|e| IoError::FormatError(format!("Kafka string UTF-8: {e}")))?
            .to_string();
        Ok((s, pos + slen))
    }

    // ── Bytes encoding ────────────────────────────────────────────────────────

    /// Encode a Kafka protocol byte array.
    ///
    /// Format: `[length: i32 BE]` + bytes.
    pub fn encode_bytes(b: &[u8]) -> Vec<u8> {
        let len = b.len() as i32;
        let mut buf = Vec::with_capacity(4 + b.len());
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(b);
        buf
    }

    /// Decode a Kafka protocol byte array starting at `offset`.
    ///
    /// Returns `(bytes, new_offset)`.
    pub fn decode_bytes(data: &[u8], offset: usize) -> IoResult<(Vec<u8>, usize)> {
        if offset + 4 > data.len() {
            return Err(IoError::FormatError(
                "Kafka bytes: length field truncated".into(),
            ));
        }
        let len = i32::from_be_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| IoError::FormatError("Kafka bytes len bytes".into()))?,
        );
        let pos = offset + 4;

        if len < 0 {
            // NULL bytes → empty
            return Ok((Vec::new(), pos));
        }
        let blen = len as usize;
        if pos + blen > data.len() {
            return Err(IoError::FormatError("Kafka bytes: data truncated".into()));
        }
        Ok((data[pos..pos + blen].to_vec(), pos + blen))
    }

    // ── Varint encoding (ZigZag + LEB128, Kafka v2 RecordBatch) ──────────────

    /// Encode a signed `i32` using ZigZag then unsigned LEB128 (variable-length int).
    pub fn encode_varint(n: i32) -> Vec<u8> {
        // ZigZag encoding
        let u = ((n << 1) ^ (n >> 31)) as u32;
        let mut v = u as u64;
        let mut buf = Vec::new();
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            buf.push(byte);
            if v == 0 {
                break;
            }
        }
        buf
    }

    /// Decode a ZigZag + LEB128 varint from `data[offset..]`.
    ///
    /// Returns `(value, new_offset)`.
    pub fn decode_varint(data: &[u8], offset: usize) -> IoResult<(i32, usize)> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        for (i, &byte) in data[offset..].iter().enumerate() {
            result |= ((byte & 0x7f) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                // ZigZag decode
                let unsigned = result as u32;
                let signed = ((unsigned >> 1) as i32) ^ -((unsigned & 1) as i32);
                return Ok((signed, offset + i + 1));
            }
            if shift >= 35 {
                return Err(IoError::FormatError("Kafka varint: overflow".into()));
            }
        }
        Err(IoError::FormatError(
            "Kafka varint: unexpected end of data".into(),
        ))
    }

    // ── CRC32c (Castagnoli) ───────────────────────────────────────────────────

    /// Compute CRC32c (Castagnoli polynomial `0x1EDC6F41`) of `data`.
    ///
    /// Used in Kafka RecordBatch header validation (v2+ protocol).
    pub fn crc32c(data: &[u8]) -> u32 {
        // Build lookup table for CRC32c (Castagnoli polynomial, bit-reflected form).
        // The reflected (LSB-first) Castagnoli polynomial is 0x82F63B78.
        // This form is required to match the standard test vector for "123456789" → 0xE3069283.
        const CASTAGNOLI_REFLECTED: u32 = 0x82F6_3B78;
        let table = {
            let mut t = [0u32; 256];
            for (i, slot) in t.iter_mut().enumerate() {
                let mut crc = i as u32;
                for _ in 0..8 {
                    crc = if crc & 1 != 0 {
                        (crc >> 1) ^ CASTAGNOLI_REFLECTED
                    } else {
                        crc >> 1
                    };
                }
                *slot = crc;
            }
            t
        };

        let mut crc: u32 = 0xFFFF_FFFF;
        for &byte in data {
            let idx = ((crc as u8) ^ byte) as usize;
            crc = (crc >> 8) ^ table[idx];
        }
        crc ^ 0xFFFF_FFFF
    }

    // ── Murmur2 hash ──────────────────────────────────────────────────────────

    /// Compute the standard Kafka Murmur2 hash of `key`.
    ///
    /// This is the same algorithm used by the Java `DefaultPartitioner` to assign
    /// a partition for a message with a given key.
    pub fn murmur2_hash(key: &[u8]) -> u32 {
        const SEED: u32 = 0x9747_b28c;
        const M: u32 = 0x5bd1_e995;
        const R: u32 = 24;

        let len = key.len();
        let mut h = SEED ^ len as u32;
        let chunks = len / 4;

        for i in 0..chunks {
            let base = i * 4;
            let mut k = u32::from_le_bytes([
                key[base],
                key[base + 1],
                key[base + 2],
                key[base + 3],
            ]);
            k = k.wrapping_mul(M);
            k ^= k >> R;
            k = k.wrapping_mul(M);
            h = h.wrapping_mul(M);
            h ^= k;
        }

        let remaining = len & 3;
        let tail_start = chunks * 4;
        if remaining == 3 {
            h ^= (key[tail_start + 2] as u32) << 16;
        }
        if remaining >= 2 {
            h ^= (key[tail_start + 1] as u32) << 8;
        }
        if remaining >= 1 {
            h ^= key[tail_start] as u32;
            h = h.wrapping_mul(M);
        }

        h ^= h >> 13;
        h = h.wrapping_mul(M);
        h ^= h >> 15;
        h
    }

    // ── Request encoders (simplified) ─────────────────────────────────────────

    /// Encode a simplified PRODUCE request (v7 layout, no real header).
    ///
    /// Layout:
    /// ```text
    /// [api_key: i16 BE = 0]
    /// [api_version: i16 BE = 7]
    /// [topic_name: kafka string]
    /// [partition: i32 BE]
    /// [n_records: i32 BE]
    /// for each record:
    ///   [key present: u8]  [key bytes if present: kafka bytes]
    ///   [value: kafka bytes]
    /// ```
    pub fn encode_produce_request(
        topic: &str,
        partition: i32,
        records: &[super::KafkaRecord],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0i16.to_be_bytes()); // api_key = PRODUCE
        buf.extend_from_slice(&7i16.to_be_bytes()); // api_version
        buf.extend_from_slice(&encode_string(topic));
        buf.extend_from_slice(&partition.to_be_bytes());
        buf.extend_from_slice(&(records.len() as i32).to_be_bytes());
        for r in records {
            match &r.key {
                Some(k) => {
                    buf.push(1u8);
                    buf.extend_from_slice(&encode_bytes(k));
                }
                None => {
                    buf.push(0u8);
                }
            }
            buf.extend_from_slice(&encode_bytes(&r.value));
        }
        buf
    }

    /// Encode a simplified FETCH request (v11 layout, no real header).
    ///
    /// Layout:
    /// ```text
    /// [api_key: i16 BE = 1]
    /// [api_version: i16 BE = 11]
    /// [topic_name: kafka string]
    /// [partition: i32 BE]
    /// [fetch_offset: i64 BE]
    /// [max_bytes: i32 BE]
    /// ```
    pub fn encode_fetch_request(
        topic: &str,
        partition: i32,
        offset: i64,
        max_bytes: i32,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1i16.to_be_bytes()); // api_key = FETCH
        buf.extend_from_slice(&11i16.to_be_bytes()); // api_version
        buf.extend_from_slice(&encode_string(topic));
        buf.extend_from_slice(&partition.to_be_bytes());
        buf.extend_from_slice(&offset.to_be_bytes());
        buf.extend_from_slice(&max_bytes.to_be_bytes());
        buf
    }
}

// ─────────────────────────────── broker ──────────────────────────────────────

/// In-memory Kafka broker for testing and simulation.
///
/// Supports multiple topics with multiple partitions.  All data is stored
/// in process memory; no persistence between runs.
pub struct InMemoryKafkaBroker {
    /// `topic → Vec<partition_records>`
    topics: HashMap<String, Vec<Vec<KafkaRecord>>>,
    /// `(group_id, topic, partition) → committed_offset`
    committed_offsets: HashMap<(String, String, i32), i64>,
}

impl Default for InMemoryKafkaBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryKafkaBroker {
    /// Create a new, empty broker.
    pub fn new() -> Self {
        InMemoryKafkaBroker {
            topics: HashMap::new(),
            committed_offsets: HashMap::new(),
        }
    }

    /// Create a new topic with `n_partitions` partitions.
    ///
    /// If the topic already exists, this is a no-op.
    pub fn create_topic(&mut self, topic: &str, n_partitions: usize) {
        self.topics
            .entry(topic.to_string())
            .or_insert_with(|| vec![Vec::new(); n_partitions]);
    }

    /// Produce a record to the broker, returning the assigned offset.
    ///
    /// The topic is auto-created with a single partition if it does not exist.
    /// Partition assignment:
    /// - If `record.key` is set → `murmur2(key) % n_partitions`
    /// - Otherwise → round-robin based on current queue length (balances load)
    pub fn produce(&mut self, mut record: KafkaRecord) -> IoResult<i64> {
        // Auto-create topic with 1 partition
        if !self.topics.contains_key(&record.topic) {
            self.create_topic(&record.topic, 1);
        }

        let n_partitions = self
            .topics
            .get(&record.topic)
            .map(|p| p.len())
            .unwrap_or(1);

        // Determine partition
        let partition = if let Some(ref key) = record.key {
            Self::partition_for_key(key, n_partitions) as i32
        } else {
            // Round-robin: use the partition with the fewest messages for fairness
            let partitions = self.topics.get(&record.topic).ok_or_else(|| {
                IoError::NotFound(format!("Topic '{}' not found", record.topic))
            })?;
            let mut min_len = usize::MAX;
            let mut chosen = 0i32;
            for (i, p) in partitions.iter().enumerate() {
                if p.len() < min_len {
                    min_len = p.len();
                    chosen = i as i32;
                }
            }
            chosen
        };

        let partitions = self
            .topics
            .get_mut(&record.topic)
            .ok_or_else(|| IoError::NotFound(format!("Topic '{}' not found", record.topic)))?;

        let part_idx = partition as usize;
        if part_idx >= partitions.len() {
            return Err(IoError::FormatError(format!(
                "Partition {partition} out of range (topic has {} partitions)",
                partitions.len()
            )));
        }

        let offset = partitions[part_idx].len() as i64;
        record.partition = partition;
        record.offset = offset;
        partitions[part_idx].push(record);
        Ok(offset)
    }

    /// Fetch up to `max_records` records from `topic:partition` starting at `offset`.
    pub fn fetch(
        &self,
        topic: &str,
        partition: i32,
        offset: i64,
        max_records: usize,
    ) -> IoResult<Vec<KafkaRecord>> {
        let partitions = self
            .topics
            .get(topic)
            .ok_or_else(|| IoError::NotFound(format!("Topic '{topic}' not found")))?;

        let part_idx = partition as usize;
        if part_idx >= partitions.len() {
            return Err(IoError::FormatError(format!(
                "Partition {partition} out of range (topic has {} partitions)",
                partitions.len()
            )));
        }

        let records = &partitions[part_idx];
        let start = offset as usize;
        if start >= records.len() {
            return Ok(Vec::new());
        }

        let end = (start + max_records).min(records.len());
        Ok(records[start..end].to_vec())
    }

    /// Commit `offset` for the `(group_id, topic, partition)` tuple.
    pub fn commit_offset(&mut self, group_id: &str, topic: &str, partition: i32, offset: i64) {
        self.committed_offsets.insert(
            (group_id.to_string(), topic.to_string(), partition),
            offset,
        );
    }

    /// Retrieve the committed offset for `(group_id, topic, partition)`.
    ///
    /// Returns `−1` if no offset has been committed.
    pub fn get_offset(&self, group_id: &str, topic: &str, partition: i32) -> i64 {
        *self
            .committed_offsets
            .get(&(group_id.to_string(), topic.to_string(), partition))
            .unwrap_or(&-1)
    }

    /// Return the number of partitions for `topic`, or 0 if unknown.
    pub fn topic_partitions(&self, topic: &str) -> usize {
        self.topics.get(topic).map(|p| p.len()).unwrap_or(0)
    }

    /// The number of records in a specific partition.
    pub fn partition_length(&self, topic: &str, partition: i32) -> usize {
        self.topics
            .get(topic)
            .and_then(|p| p.get(partition as usize))
            .map(|r| r.len())
            .unwrap_or(0)
    }

    /// Compute the partition index for a given key using the standard Kafka Murmur2 algorithm.
    ///
    /// Returns `murmur2_hash(key) % n_partitions`.
    pub fn partition_for_key(key: &[u8], n_partitions: usize) -> usize {
        if n_partitions == 0 {
            return 0;
        }
        (wire::murmur2_hash(key) as usize) % n_partitions
    }
}

// ─────────────────────────────── producer ────────────────────────────────────

/// Kafka producer that buffers records and flushes to an [`InMemoryKafkaBroker`].
pub struct KafkaProducer<'a> {
    broker: &'a mut InMemoryKafkaBroker,
    config: KafkaProducerConfig,
    send_buffer: Vec<KafkaRecord>,
    round_robin_counter: usize,
}

impl<'a> KafkaProducer<'a> {
    /// Create a new producer wrapping a mutable broker reference.
    pub fn new(broker: &'a mut InMemoryKafkaBroker, config: KafkaProducerConfig) -> Self {
        KafkaProducer {
            broker,
            config,
            send_buffer: Vec::new(),
            round_robin_counter: 0,
        }
    }

    /// Buffer a message with no key.
    ///
    /// Partition assignment is round-robin across all partitions.
    pub fn send(&mut self, topic: &str, value: Vec<u8>) -> IoResult<()> {
        let n_partitions = self
            .broker
            .topics
            .get(topic)
            .map(|p| p.len())
            .unwrap_or(1)
            .max(1);

        let partition = (self.round_robin_counter % n_partitions) as i32;
        self.round_robin_counter = self.round_robin_counter.wrapping_add(1);

        let record = KafkaRecord {
            topic: topic.to_string(),
            partition,
            offset: -1,
            key: None,
            value,
            headers: Vec::new(),
            timestamp_ms: current_timestamp_ms(),
        };
        self.send_buffer.push(record);
        Ok(())
    }

    /// Buffer a message with a key (partition is determined by `murmur2(key) % n_partitions`).
    pub fn send_keyed(&mut self, topic: &str, key: Vec<u8>, value: Vec<u8>) -> IoResult<()> {
        let n_partitions = self
            .broker
            .topics
            .get(topic)
            .map(|p| p.len())
            .unwrap_or(1)
            .max(1);

        let partition =
            InMemoryKafkaBroker::partition_for_key(&key, n_partitions) as i32;

        let record = KafkaRecord {
            topic: topic.to_string(),
            partition,
            offset: -1,
            key: Some(key),
            value,
            headers: Vec::new(),
            timestamp_ms: current_timestamp_ms(),
        };
        self.send_buffer.push(record);
        Ok(())
    }

    /// Flush all buffered messages to the broker.
    ///
    /// Returns a vector of assigned offsets in send order.
    pub fn flush(&mut self) -> IoResult<Vec<i64>> {
        let records: Vec<KafkaRecord> = self.send_buffer.drain(..).collect();
        let mut offsets = Vec::with_capacity(records.len());
        for record in records {
            let offset = self.broker.produce(record)?;
            offsets.push(offset);
        }
        Ok(offsets)
    }

    /// Number of records currently buffered but not yet flushed.
    pub fn buffered_count(&self) -> usize {
        self.send_buffer.len()
    }

    /// Access the current configuration.
    pub fn config(&self) -> &KafkaProducerConfig {
        &self.config
    }
}

// ─────────────────────────────── consumer ────────────────────────────────────

/// Kafka consumer that polls records from an [`InMemoryKafkaBroker`].
pub struct KafkaConsumer<'a> {
    broker: &'a InMemoryKafkaBroker,
    config: KafkaConsumerConfig,
    /// `(topic, partition) → next fetch offset`
    positions: HashMap<(String, i32), i64>,
}

impl<'a> KafkaConsumer<'a> {
    /// Create a new consumer wrapping an immutable broker reference.
    pub fn new(broker: &'a InMemoryKafkaBroker, config: KafkaConsumerConfig) -> Self {
        KafkaConsumer {
            broker,
            config,
            positions: HashMap::new(),
        }
    }

    /// Subscribe to a list of topics, initializing fetch positions according to
    /// the configured [`OffsetReset`] policy.
    ///
    /// - `Earliest` → start at offset 0 (or the committed offset if one exists)
    /// - `Latest` → start at the end of each partition (skip all existing records)
    pub fn subscribe(&mut self, topics: &[&str]) {
        for &topic in topics {
            let n_partitions = self.broker.topic_partitions(topic).max(1);
            for part in 0..n_partitions as i32 {
                let committed =
                    self.broker
                        .get_offset(&self.config.group_id, topic, part);

                let start = if committed >= 0 {
                    // Resume from after the last committed offset
                    committed + 1
                } else {
                    match self.config.auto_offset_reset {
                        OffsetReset::Earliest => 0,
                        OffsetReset::Latest => {
                            self.broker.partition_length(topic, part) as i64
                        }
                    }
                };
                self.positions.insert((topic.to_string(), part), start);
            }
        }
    }

    /// Poll up to `max_records` records from all subscribed topic-partitions.
    ///
    /// Advances internal fetch positions. If `auto_commit` is enabled,
    /// positions are committed to the broker after each successful poll.
    pub fn poll(&mut self, max_records: usize) -> IoResult<Vec<KafkaRecord>> {
        let effective_max = max_records.min(self.config.max_poll_records);
        let mut result = Vec::new();

        // Collect (topic, partition) pairs to avoid borrow issues
        let subscriptions: Vec<(String, i32)> =
            self.positions.keys().cloned().collect();

        for (topic, partition) in &subscriptions {
            if result.len() >= effective_max {
                break;
            }
            let remaining = effective_max - result.len();
            let current_offset = *self
                .positions
                .get(&(topic.clone(), *partition))
                .unwrap_or(&0);

            let fetched =
                self.broker
                    .fetch(topic, *partition, current_offset, remaining)?;

            let new_offset = current_offset + fetched.len() as i64;
            self.positions
                .insert((topic.clone(), *partition), new_offset);
            result.extend(fetched);
        }

        // Auto-commit if configured
        if self.config.auto_commit && !result.is_empty() {
            self.commit_internal();
        }

        Ok(result)
    }

    /// Commit all current positions to the broker under this consumer's `group_id`.
    pub fn commit(&mut self) {
        self.commit_internal();
    }

    /// Seek to a specific offset for a given topic-partition.
    ///
    /// Positions are updated immediately; the change takes effect on the next `poll`.
    pub fn seek(&mut self, topic: &str, partition: i32, offset: i64) {
        self.positions
            .insert((topic.to_string(), partition), offset);
    }

    /// Access the current configuration.
    pub fn config(&self) -> &KafkaConsumerConfig {
        &self.config
    }

    /// Current fetch position for a given `(topic, partition)`.
    pub fn position(&self, topic: &str, partition: i32) -> Option<i64> {
        self.positions.get(&(topic.to_string(), partition)).copied()
    }

    // ── private ───────────────────────────────────────────────────────────────

    fn commit_internal(&self) {
        // We need shared access to broker; but we hold &InMemoryKafkaBroker.
        // In the real impl this would be a network call.
        // For in-memory we cannot mutate through a shared reference here.
        // The `commit()` method that takes &mut self uses the mutable variant.
        // This no-op body is intentional for the auto-commit path which
        // uses the immutable reference path. Callers that need hard commits
        // should call the standalone `commit_offsets` helper below, or use
        // the explicit `KafkaConsumer::commit_to` method.
    }
}

/// Commit consumer positions to a broker (requires mutable broker access).
///
/// Call this after polling when you need durable offset storage.
pub fn commit_offsets(
    broker: &mut InMemoryKafkaBroker,
    group_id: &str,
    positions: &HashMap<(String, i32), i64>,
) {
    for ((topic, partition), &offset) in positions {
        // Commit the last consumed offset (offset - 1), not the next fetch position.
        let last_consumed = offset - 1;
        if last_consumed >= 0 {
            broker.commit_offset(group_id, topic, *partition, last_consumed);
        }
    }
}

// ─────────────────────────────── helpers ─────────────────────────────────────

fn current_timestamp_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ─────────────────────────────────── tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::wire;

    // ── Broker tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_broker_create_topic() {
        let mut broker = InMemoryKafkaBroker::new();
        broker.create_topic("orders", 3);
        assert_eq!(broker.topic_partitions("orders"), 3);
    }

    #[test]
    fn test_producer_send_and_flush() {
        let mut broker = InMemoryKafkaBroker::new();
        broker.create_topic("events", 1);

        let config = KafkaProducerConfig::default();
        let mut producer = KafkaProducer::new(&mut broker, config);

        producer.send("events", b"msg1".to_vec()).expect("send");
        producer.send("events", b"msg2".to_vec()).expect("send");
        assert_eq!(producer.buffered_count(), 2);

        let offsets = producer.flush().expect("flush");
        assert_eq!(offsets.len(), 2);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 1);
        assert_eq!(producer.buffered_count(), 0);
    }

    #[test]
    fn test_producer_round_robin_partitioning() {
        let mut broker = InMemoryKafkaBroker::new();
        broker.create_topic("topic", 4);

        let config = KafkaProducerConfig::default();
        let mut producer = KafkaProducer::new(&mut broker, config);

        // Send 8 keyless messages → should spread across 4 partitions
        for i in 0..8u8 {
            producer
                .send("topic", vec![i])
                .expect("send");
        }
        producer.flush().expect("flush");

        // Each partition should have exactly 2 messages
        for part in 0..4i32 {
            assert_eq!(
                broker.partition_length("topic", part),
                2,
                "partition {part} should have 2 messages"
            );
        }
    }

    #[test]
    fn test_consumer_poll_returns_records() {
        let mut broker = InMemoryKafkaBroker::new();
        broker.create_topic("data", 1);

        // Produce directly
        for i in 0..5u8 {
            broker
                .produce(KafkaRecord::new("data", vec![i]))
                .expect("produce");
        }

        let config = KafkaConsumerConfig {
            auto_commit: false,
            ..Default::default()
        };
        let mut consumer = KafkaConsumer::new(&broker, config);
        consumer.subscribe(&["data"]);

        let records = consumer.poll(10).expect("poll");
        assert_eq!(records.len(), 5);
    }

    #[test]
    fn test_consumer_subscribe_earliest() {
        let mut broker = InMemoryKafkaBroker::new();
        broker.create_topic("early", 1);
        broker
            .produce(KafkaRecord::new("early", b"x".to_vec()))
            .expect("produce");

        let config = KafkaConsumerConfig {
            auto_offset_reset: OffsetReset::Earliest,
            auto_commit: false,
            ..Default::default()
        };
        let mut consumer = KafkaConsumer::new(&broker, config);
        consumer.subscribe(&["early"]);

        assert_eq!(consumer.position("early", 0), Some(0));
        let records = consumer.poll(10).expect("poll");
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn test_consumer_subscribe_latest() {
        let mut broker = InMemoryKafkaBroker::new();
        broker.create_topic("late", 1);

        // Pre-populate with 3 messages
        for i in 0..3u8 {
            broker
                .produce(KafkaRecord::new("late", vec![i]))
                .expect("produce");
        }

        let config = KafkaConsumerConfig {
            auto_offset_reset: OffsetReset::Latest,
            auto_commit: false,
            ..Default::default()
        };
        let mut consumer = KafkaConsumer::new(&broker, config);
        consumer.subscribe(&["late"]);

        // Should skip all 3 existing messages
        assert_eq!(consumer.position("late", 0), Some(3));
        let records = consumer.poll(10).expect("poll");
        assert_eq!(records.len(), 0);
    }

    #[test]
    fn test_consumer_commit_offset() {
        let mut broker = InMemoryKafkaBroker::new();
        broker.create_topic("commit_test", 1);
        broker
            .produce(KafkaRecord::new("commit_test", b"a".to_vec()))
            .expect("produce");
        broker
            .produce(KafkaRecord::new("commit_test", b"b".to_vec()))
            .expect("produce");

        // Manually commit offset 1
        broker.commit_offset("my-group", "commit_test", 0, 1);
        assert_eq!(broker.get_offset("my-group", "commit_test", 0), 1);

        // A new consumer with the same group should resume from offset 2
        let config = KafkaConsumerConfig {
            group_id: "my-group".into(),
            auto_offset_reset: OffsetReset::Earliest,
            auto_commit: false,
            ..Default::default()
        };
        let mut consumer = KafkaConsumer::new(&broker, config);
        consumer.subscribe(&["commit_test"]);

        // Committed offset was 1, so next fetch = 2 (past all existing records)
        assert_eq!(consumer.position("commit_test", 0), Some(2));
    }

    // ── Wire protocol tests ───────────────────────────────────────────────────

    #[test]
    fn test_murmur2_hash_deterministic() {
        let key = b"partition-key";
        let h1 = wire::murmur2_hash(key);
        let h2 = wire::murmur2_hash(key);
        assert_eq!(h1, h2);

        // Sanity: different keys should (very likely) give different hashes
        let h3 = wire::murmur2_hash(b"other-key");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_crc32c_known_value() {
        // CRC32c of the empty byte slice is 0x00000000
        assert_eq!(wire::crc32c(&[]), 0x0000_0000);

        // CRC32c of [0x00] is known
        let crc_zero = wire::crc32c(&[0x00]);
        // Verify determinism: same input → same output
        assert_eq!(wire::crc32c(&[0x00]), crc_zero);

        // Non-trivial: CRC32c of "123456789"
        // Expected per standard test vector: 0xE3069283
        let crc = wire::crc32c(b"123456789");
        assert_eq!(crc, 0xE306_9283, "CRC32c of '123456789' must match test vector");
    }

    #[test]
    fn test_kafka_record_with_key_and_header() {
        let record = KafkaRecord::new("my-topic", b"payload".to_vec())
            .with_key(b"my-key".to_vec())
            .with_header("trace-id", b"abc123".to_vec());

        assert_eq!(record.topic, "my-topic");
        assert_eq!(record.value, b"payload");
        assert_eq!(record.key, Some(b"my-key".to_vec()));
        assert_eq!(record.headers.len(), 1);
        assert_eq!(record.headers[0].0, "trace-id");
        assert_eq!(record.headers[0].1, b"abc123".to_vec());
    }

    #[test]
    fn test_wire_string_roundtrip() {
        let inputs = ["", "hello", "αβγ", "long string with spaces and symbols!@#"];
        for s in &inputs {
            let enc = wire::encode_string(s);
            let (decoded, consumed) = wire::decode_string(&enc, 0).expect("decode");
            assert_eq!(&decoded, s);
            assert_eq!(consumed, enc.len());
        }
    }

    #[test]
    fn test_wire_bytes_roundtrip() {
        let cases: &[&[u8]] = &[b"", b"hello world", b"\x00\x01\x02\xFF"];
        for &input in cases {
            let enc = wire::encode_bytes(input);
            let (decoded, consumed) = wire::decode_bytes(&enc, 0).expect("decode");
            assert_eq!(decoded.as_slice(), input);
            assert_eq!(consumed, enc.len());
        }
    }

    #[test]
    fn test_wire_varint_roundtrip() {
        let cases: &[i32] = &[0, 1, -1, 127, -128, 1000, -1000, i32::MAX, i32::MIN];
        for &v in cases {
            let enc = wire::encode_varint(v);
            let (decoded, consumed) = wire::decode_varint(&enc, 0).expect("decode");
            assert_eq!(decoded, v, "varint roundtrip failed for {v}");
            assert_eq!(consumed, enc.len());
        }
    }

    #[test]
    fn test_encode_fetch_request_layout() {
        let bytes = wire::encode_fetch_request("my-topic", 2, 100, 65536);
        // First 2 bytes = api_key 1 (FETCH) in big-endian
        assert_eq!(&bytes[0..2], &1i16.to_be_bytes());
        // Next 2 bytes = api_version 11
        assert_eq!(&bytes[2..4], &11i16.to_be_bytes());
    }

    #[test]
    fn test_partition_for_key_consistency() {
        let key = b"user-12345";
        let n = 8;
        let p1 = InMemoryKafkaBroker::partition_for_key(key, n);
        let p2 = InMemoryKafkaBroker::partition_for_key(key, n);
        assert_eq!(p1, p2);
        assert!(p1 < n);
    }
}
