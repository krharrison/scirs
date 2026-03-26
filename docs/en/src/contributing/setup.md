# Development Setup

## Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs))
- Git
- Approximately 4 GB of disk space for the full build

No C compiler, Fortran compiler, or system libraries are required. SciRS2 is pure Rust.

## Cloning and Building

```bash
git clone https://github.com/cool-japan/scirs.git
cd scirs

# Check the entire workspace compiles
cargo check --all-features --workspace

# Build in release mode (optimized)
cargo build --release --workspace
```

## Running Tests

SciRS2 uses [cargo-nextest](https://nexte.st/) for test execution:

```bash
# Install nextest
cargo install cargo-nextest

# Run all tests (excluding Python bindings and heavy dataset examples)
cargo nextest run --all-features --workspace \
    --exclude scirs2-python \
    --exclude scirs2-datasets

# Run tests for a specific crate
cargo nextest run -p scirs2-linalg --all-features

# Run a specific test by name
cargo nextest run -p scirs2-stats test_normal_distribution
```

The `scirs2-python` crate is excluded because it requires a Python development
environment. The `scirs2-datasets` crate is excluded from full runs because its
examples can cause linker OOM on some systems; use `--lib` to run just its unit tests:

```bash
cargo nextest run -p scirs2-datasets --lib
```

## Formatting and Linting

```bash
# Format all code
cargo fmt --all

# Run clippy with workspace lints
cargo clippy --all-features --workspace

# Check documentation builds without warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

## Benchmarks

```bash
# Run the benchmark regression script
./scripts/bench-regression.sh

# Run benchmarks for a specific crate
cargo bench -p scirs2-linalg
cargo bench -p scirs2-fft
```

## Miri (Undefined Behavior Checks)

```bash
# Install miri
rustup component add miri

# Run miri on core crates
cargo miri test -p scirs2-core -- --test-threads=1
```

## Documentation

Build and preview this book locally:

```bash
# Install mdbook
cargo install mdbook

# Build and serve
cd docs/en
mdbook serve --open
```

Build the API reference:

```bash
cargo doc --workspace --no-deps --open
```

## Policy Checker

SciRS2 ships a custom Cargo subcommand for verifying project policies:

```bash
cargo run -p cargo-scirs2-policy -- check --workspace .
```

This checks for:
- No `unwrap()` calls in production code
- No C/Fortran dependencies in default features
- Version consistency across workspace members
- Naming conventions (snake_case for variables, etc.)
