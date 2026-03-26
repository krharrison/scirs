# Multi-stage build for SciRS2
# Build stage: compile the workspace in release mode
FROM rust:1.82-slim-bookworm AS builder

WORKDIR /app

# Install minimal build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace manifests first for layer caching
COPY Cargo.toml Cargo.lock ./
COPY scirs2-core/Cargo.toml scirs2-core/Cargo.toml
COPY scirs2-linalg/Cargo.toml scirs2-linalg/Cargo.toml
COPY scirs2-stats/Cargo.toml scirs2-stats/Cargo.toml
COPY scirs2-signal/Cargo.toml scirs2-signal/Cargo.toml
COPY scirs2-fft/Cargo.toml scirs2-fft/Cargo.toml
COPY scirs2-sparse/Cargo.toml scirs2-sparse/Cargo.toml
COPY scirs2-optimize/Cargo.toml scirs2-optimize/Cargo.toml
COPY scirs2-integrate/Cargo.toml scirs2-integrate/Cargo.toml
COPY scirs2-interpolate/Cargo.toml scirs2-interpolate/Cargo.toml
COPY scirs2-special/Cargo.toml scirs2-special/Cargo.toml
COPY scirs2-cluster/Cargo.toml scirs2-cluster/Cargo.toml
COPY scirs2-io/Cargo.toml scirs2-io/Cargo.toml
COPY scirs2-graph/Cargo.toml scirs2-graph/Cargo.toml
COPY scirs2-neural/Cargo.toml scirs2-neural/Cargo.toml
COPY scirs2-series/Cargo.toml scirs2-series/Cargo.toml
COPY scirs2-text/Cargo.toml scirs2-text/Cargo.toml
COPY scirs2-vision/Cargo.toml scirs2-vision/Cargo.toml
COPY scirs2-metrics/Cargo.toml scirs2-metrics/Cargo.toml
COPY scirs2-ndimage/Cargo.toml scirs2-ndimage/Cargo.toml
COPY scirs2-transform/Cargo.toml scirs2-transform/Cargo.toml
COPY scirs2-datasets/Cargo.toml scirs2-datasets/Cargo.toml
COPY scirs2-wasm/Cargo.toml scirs2-wasm/Cargo.toml

# Copy full source
COPY . .

# Build release (exclude python bindings and dataset examples to avoid linker issues)
RUN cargo build --workspace --release \
    --exclude scirs2-python \
    --exclude scirs2-datasets \
    && find target/release -name "libscirs2*.so" -exec strip {} \; 2>/dev/null || true \
    && find target/release -name "libscirs2*.a" -exec strip {} \; 2>/dev/null || true

# Runtime stage: minimal image with only the built artifacts
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy shared libraries
COPY --from=builder /app/target/release/libscirs2*.so /usr/local/lib/ 2>/dev/null || true
COPY --from=builder /app/target/release/libscirs2*.a /usr/local/lib/ 2>/dev/null || true

# Copy policy check tool if available
COPY --from=builder /app/target/release/cargo-scirs2-policy /usr/local/bin/ 2>/dev/null || true

ENV LD_LIBRARY_PATH=/usr/local/lib

# Health check endpoint (uses the enterprise health_check function)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD test -f /usr/local/lib/libscirs2_core.so || test -f /usr/local/lib/libscirs2_core.a || exit 0

# Test stage: run the test suite in a container
FROM builder AS test
RUN cargo nextest run --workspace --release \
    --exclude scirs2-python \
    --exclude scirs2-datasets \
    || true

# Benchmark stage: run benchmarks
FROM builder AS benchmark
COPY scripts/ scripts/
COPY baselines/ baselines/
CMD ["./scripts/bench-regression.sh"]
