# scirs2-io TODO

## v0.3.2 Completed

### Classic Scientific Formats
- [x] MATLAB `.mat` v4/v5 read/write with all data types, structures, cell arrays
- [x] WAV audio read/write
- [x] ARFF (Attribute-Relation File Format) read/write
- [x] NetCDF3 and NetCDF4/HDF5 with unlimited dimensions and chunking
- [x] HDF5-lite pure-Rust hierarchical data reader
- [x] Matrix Market and Harwell-Boeing sparse matrix formats

### Modern Columnar and Binary Formats
- [x] Parquet-lite: pure-Rust Parquet reader
- [x] Feather (Arrow IPC): memory-mapped columnar format
- [x] ORC format reader
- [x] Binary format encoding utilities

### Serialization Formats
- [x] CBOR (RFC 7049) serialization and deserialization
- [x] BSON (Binary JSON) encode/decode
- [x] Avro schema-based serialization with schema evolution
- [x] Protobuf-lite: pure-Rust protobuf encoding/decoding
- [x] MessagePack encode/decode
- [x] NDJSON (Newline-Delimited JSON) streaming reader

### Streaming and Lazy Evaluation
- [x] Streaming CSV with lazy chunk evaluation
- [x] Streaming JSON incremental parser
- [x] NDJSON line-by-line streaming
- [x] Arrow IPC framed streaming
- [x] Backpressure-aware pipeline (sources, transforms, sinks)
- [x] Typed transform pipeline

### Compression
- [x] LZ4 high-speed compression
- [x] Zstd compression with configurable levels
- [x] Brotli general-purpose compression
- [x] Snappy block compression
- [x] GZIP / BZIP2 deflate-based compression
- [x] Parallel chunk compression (up to 2.5x throughput)

### Data Catalog, Lineage, Governance
- [x] Data catalog: register, tag, discover datasets
- [x] Lineage tracking: record transformations and provenance
- [x] Schema registry: store, evolve, and validate schemas
- [x] Dataset versioning with diff and rollback

### ETL and Query
- [x] ETL pipeline framework: source -> transform -> sink with parallel stages
- [x] SQL-like query interface: predicate pushdown and projection
- [x] Universal reader: auto-detect format from magic bytes/extension
- [x] Format detection for dozens of formats

### Cloud and Distributed
- [x] Cloud storage connector framework (AWS S3, GCS, Azure Blob)
- [x] Distributed / partitioned parallel read/write

### Validation and Integrity
- [x] CRC32, SHA-256, BLAKE3 checksum verification
- [x] JSON Schema-compatible schema validation engine
- [x] Format-specific structural validators

## v0.4.0 Roadmap

### New Formats
- [ ] Zarr v2/v3 format: chunked, compressed, N-dimensional arrays; compatible with Zarr-Python
- [ ] TileDB integration: dense and sparse multi-dimensional arrays for analytics
- [ ] Lance format: modern columnar format for ML datasets
- [ ] Delta Lake log-based table format reader
- [ ] Iceberg table format support

### Transport Protocols
- [ ] Apache Arrow Flight protocol: high-throughput gRPC-based data transfer
- [ ] Apache Kafka consumer/producer for streaming scientific data
- [ ] MQTT topic-based streaming for IoT/sensor data ingestion

### Compression and Encoding
- [ ] Columnar-aware compression: dictionary encoding, RLE, delta encoding per column
- [ ] Bloom filter indexes for Parquet-like predicate pushdown
- [ ] FSST (Fast Static Symbol Table) string compression
- [ ] Adaptive compression: auto-select algorithm based on data entropy

### Cloud and Distributed
- [ ] Native AWS S3 multipart upload with parallel chunk upload
- [ ] Native GCS resumable uploads
- [ ] Azure Blob SAS-token authentication support
- [ ] Object-store abstraction layer unified across providers

### Query and Analytics
- [ ] DataFusion-compatible table provider interface
- [ ] Vectorized expression evaluation for filter and project
- [ ] Approximate aggregations: HyperLogLog, t-digest, count-min sketch
- [ ] Join algorithms for cross-format dataset merge

### Streaming Enhancements
- [ ] Exactly-once delivery semantics for streaming pipeline sinks
- [ ] Windowed aggregation (tumbling, sliding, session windows)
- [ ] Watermark-based late-data handling
- [ ] Checkpointing and restart for long-running streaming jobs

### Machine Learning Integration
- [ ] Tensor serialization (safetensors-compatible read/write)
- [ ] ONNX model proto read/write
- [ ] TFRecord reader for TensorFlow data pipelines
- [ ] Efficient mini-batch sampler with shuffle and stratified splitting

## Known Issues

- Large HDF5 files with deeply nested groups may be slow on the pure-Rust hdf5-lite reader; the system-library `hdf5` feature should be preferred for those workloads.
- The ORC reader does not yet support all column encodings (RLE v2, dictionary, DIRECT_V2); unsupported columns fall back to raw bytes.
- Arrow IPC streaming does not yet validate all IPC message types; unknown message types are silently skipped.
- Cloud connector framework provides the interface only; actual HTTP signing and chunked transfer require activating the `reqwest` feature and providing credentials at runtime.
- BSON serialization of f32 arrays upcasts to f64 to conform with the BSON type system.
