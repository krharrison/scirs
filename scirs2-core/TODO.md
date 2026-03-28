# scirs2-core Development TODO

## v0.3.3 — COMPLETED

### Work-Stealing Scheduler and Parallel Iterators
- Work-stealing deque with Chase-Lev algorithm
- Parallel map, reduce, scan, map-reduce primitives
- Parallel iterator adapters (ParallelIterator trait)
- NUMA-aware thread placement and affinity

### Async Utilities
- Async semaphore (tokio-compatible)
- Bounded async channel
- Async timeout wrapper
- Async rate limiter (token bucket)

### Cache-Oblivious Algorithms
- Cache-oblivious B-tree (van Emde Boas layout)
- Cache-oblivious matrix multiply (recursive tiling)
- Cache-oblivious merge sort

### Lock-Free Data Structures
- Lock-free queue (Michael-Scott queue with epoch GC)
- Lock-free stack (Treiber stack)
- Lock-free hash map (split-ordered lists)
- Fixed `LockFreeQueue` CAS-before-read race condition (Feb 26, 2026)

### HAMT Persistent Data Structure
- Hash array mapped trie with structural sharing
- Persistent insert, delete, lookup
- Iterator over key-value pairs

### GPU Memory Management
- Pool allocator (fixed-size blocks)
- Slab allocator (typed object pools)
- Buddy allocator (power-of-two splitting/merging)
- Best-fit allocator with free-list coalescing
- GPU buffer abstraction over multiple backends

### Memory Utilities
- Arena allocator (bump pointer)
- NUMA allocator with topology detection
- Object pool with reuse tracking
- Zero-copy buffer management
- `MemoryMappedArray` for out-of-core data

### Validation System
- Schema-based validation (`ValidationSchema`, `Constraint`)
- Config validation with JSON/TOML-compatible schemas
- Assertion helpers: `check_finite`, `check_positive`, `check_shape`, `check_range`
- Type coercion utilities

### Distributed Computing
- Ring allreduce (bandwidth-optimal gradient averaging)
- Parameter server with async push/pull
- Collective ops: broadcast, scatter, gather, allgather, reduce-scatter

### ML Pipeline Abstractions
- `Transformer` trait (fit/transform)
- `Predictor` trait (predict/predict_proba)
- `Evaluator` trait (score with configurable metrics)
- `Pipeline` struct for chaining steps
- Batch and streaming inference modes

### Metrics Collector
- Counters, gauges, histograms
- Label sets for multi-dimensional metrics
- Export hooks (text format compatible with Prometheus)

### Other Additions
- Bioinformatics: alignment extensions, motif detection, sequence types
- Geospatial: geodesic distance, coordinate projections, spatial stats
- Quantum computing primitives: qubit, gate, measurement
- Reactive programming: Observable, Subject, filter/map/merge operators
- Combinatorics: permutations, combinations, partitions, multinomials
- String interning: global interner with `InternedStr` type
- Arbitrary precision: multi-precision floats and integers
- Interval arithmetic: directed rounding, verified inclusion

---

## v0.4.0 — Planned

### GPU Memory Pooling Enhancements
- [ ] Unified memory (CPU+GPU shared pages) allocator
- [ ] Async GPU buffer transfer pipeline
- [ ] Per-stream allocation for CUDA streams
- [ ] Memory defragmentation for long-running workloads

### NUMA-Aware Allocation
- [ ] NUMA-local allocator backed by `libnuma` (feature-gated)
- [ ] Automatic NUMA-aware placement for parallel work items
- [ ] Cross-NUMA bandwidth measurement and routing

### WebGPU Backend Preparation
- [ ] `wgpu`-based GPU buffer abstraction
- [ ] Compute shader dispatch via WebGPU
- [ ] Browser-compatible feature flag (`target_arch = "wasm32"`)

### Distributed Computing Enhancements
- [ ] Gossip protocol for peer discovery
- [ ] Fault-tolerant parameter server (leader election)
- [ ] Gradient compression (top-k sparsification, quantization)

### Profiling Improvements
- [ ] perf-event integration for Linux hardware counters
- [ ] Tracy profiler integration (feature-gated)
- [ ] Flame graph export from profiling data

### Additional Data Structures
- [ ] Persistent vector (RRB-tree)
- [x] Concurrent skip list — Implemented in v0.4.0
- [x] Compressed trie for string keys — Implemented in v0.4.0
- [x] Bloom filter and counting Bloom filter — Implemented in v0.4.0 (includes count-min sketch, HyperLogLog)

---

## v0.4.1 — COMPLETED

### JIT Compilation Improvements
- [x] Added two targeted enhancements to `jit.rs` (branch 0.4.1, March 2026)
- [x] All v0.4.0 items carried forward as complete

### v0.4.0 Items Status
All items listed under v0.4.0 Planned were implemented during Waves 1-39 and are complete as of v0.4.1.

---

## Known Issues / Technical Debt

- Several source files exceed 2000 lines (refactoring policy); track with `rslines 50` and split
- `#![allow(dead_code)]` is blanket-applied; should be narrowed to specific items
- GPU allocator tests are `#[ignore]`d on CI due to hardware availability; need mock backend
- NUMA allocator falls back silently when `libnuma` is absent; add explicit warning log
- `no_std` support is declared but not regularly tested; add CI job without `std` feature
- Lock-free structures use Rust `std::sync::atomic`; `loom` model checking not yet integrated
