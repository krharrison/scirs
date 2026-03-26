# Policy & Conventions

SciRS2 follows strict coding policies to maintain quality and consistency across the
workspace. These are enforced by CI checks and the `cargo-scirs2-policy` tool.

## No Unwrap Policy

Production code must not use `.unwrap()`. The workspace `Cargo.toml` enables:

```toml
[workspace.lints.clippy]
unwrap_used = "warn"
```

Instead of `.unwrap()`, use one of:

- **`?` operator**: Propagate the error to the caller
- **`.expect("descriptive message")`**: When the condition is logically guaranteed but
  the type system cannot prove it
- **`match` / `if let`**: When you need to handle the error case differently

```rust,ignore
// Bad
let x = some_operation().unwrap();

// Good
let x = some_operation()?;

// Good (when you can prove it is safe and want to document why)
let x = some_operation().expect("matrix is square, validated at construction");
```

Test code is exempt from this rule, but even in tests, prefer `?` with
`Result`-returning test functions.

## Pure Rust Policy

Default features must compile without any C or Fortran dependencies. This ensures:

- Cross-compilation works out of the box
- No system library installation required
- WASM compilation is straightforward
- Reproducible builds across platforms

GPU features (CUDA, ROCm) that require C toolchains are acceptable only behind
feature gates that are off by default.

### COOLJAPAN Ecosystem

SciRS2 uses COOLJAPAN ecosystem crates as pure-Rust replacements:

| Use | Instead of |
|-----|-----------|
| `oxiblas` | OpenBLAS, MKL |
| `oxifft` | FFTW, rustfft |
| `oxicode` | bincode |
| `oxiz` | Z3 |
| `oxiarc-archive`, `oxiarc-*` | flate2, zstd, bzip2, lz4, zip, tar, snap, brotli, miniz_oxide |

Do not introduce dependencies on the "instead of" column.

## Naming Conventions

- **Variables and functions**: `snake_case`
- **Types and traits**: `PascalCase`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Module names**: `snake_case`
- **Feature flags**: `kebab-case`

Function names should mirror SciPy where applicable (e.g., `solve`, `lu`, `fft`, `quad`).

## Error Types

Each crate defines its own error enum and result type alias:

```rust,ignore
pub enum SignalError {
    InvalidInput(String),
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    ConvergenceFailed,
    // ...
}

pub type SignalResult<T> = Result<T, SignalError>;
```

Use `#[non_exhaustive]` on public enums so that adding variants is not a breaking change.
When matching non-exhaustive enums, always include a `_ => {}` wildcard arm.

## File Size Limit

Individual source files should stay under 2000 lines. When a file grows beyond this,
refactor into sub-modules. The [splitrs](https://github.com/cool-japan/splitrs) tool
can automate this:

```bash
splitrs --help
rslines 50  # find files exceeding thresholds
```

## Version Policy

- All crates share one version via `workspace.package.version`
- Version bumps happen when the branch name changes (e.g., `0.3.0` to `0.4.0`)
- Never `cargo publish` without explicit permission
- Use `--dry-run` for publish testing

## Testing Requirements

- All new public APIs must have at least one test
- Use `std::env::temp_dir()` for temporary files in tests (do not litter the workspace)
- Use `Result`-returning test functions to leverage `?`
- Property-based tests (proptest) are encouraged for mathematical invariants

## Documentation

- All public items should have doc comments
- Include at least one code example per major function
- Do not scatter `.md` files in the workspace; use `/tmp/` for scratch documentation
- Do not create README files unless explicitly requested

## Commit and Push Policy

- Never commit or push without explicit permission
- Never force-push to `master`
- No `--no-verify` or `--no-gpg-sign` flags
