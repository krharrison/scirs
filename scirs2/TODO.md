# scirs2 Meta-Crate TODO

## Status: v0.3.1 Released (March 9, 2026)

## Purpose

The `scirs2` meta-crate is the all-in-one convenience entry-point for the SciRS2 ecosystem. It re-exports all sub-crates via Cargo feature flags, exposing them as unified top-level modules. Users who want a single dependency add `scirs2 = "0.3.1"` instead of listing each sub-crate individually.

---

## v0.3.1 Completed

- [x] Feature-gated re-exports for all 23 sub-crates
- [x] `standard`, `ai`, `experimental`, `full` feature groups
- [x] `oxifft` feature for high-performance pure-Rust FFT via OxiFFT
- [x] `prelude` module re-exporting common types across the ecosystem
- [x] Workspace-unified version (`version.workspace = true`)
- [x] Pure Rust by default (no C/Fortran transitive dependencies in default features)
- [x] README updated with all feature flags and quick-start examples

---

## v0.4.0 Planned

- [ ] `cuda` feature group: gates GPU-accelerated paths in sub-crates when CUDA kernels land
- [ ] `rocm` feature group: AMD GPU acceleration
- [ ] `distributed` feature: enables cluster/MPI abstractions in scirs2-core and sub-crates
- [ ] Re-export `scirs2_wasm` module under `wasm` feature for WASM builds
- [ ] `benchmarks` feature: expose benchmark helpers from scirs2-datasets
- [ ] Expand `prelude` to cover more commonly-used types from new v0.4.0 additions
- [ ] Auto-generated feature matrix in documentation (docs.rs all-features already set)

---

## v0.5.0 Planned

- [ ] `jit` feature: gates JAX-style functional transformation framework
- [ ] `vmap`/`pmap` re-exports from scirs2-autograd under top-level `scirs2::transforms`
- [ ] `mobile` feature group: iOS Metal + Android NNAPI gated paths
- [ ] Structured `scirs2::nn` re-export namespace for neural architecture types
- [ ] `symbolic` feature: planned scirs2-symbolic crate integration

---

## Ongoing Maintenance

- [ ] Keep feature list in README.md in sync with Cargo.toml on every release
- [ ] Verify that `cargo doc --all-features` renders correctly on docs.rs after each release
- [ ] Confirm `cargo check --no-default-features` and each individual feature flag compile cleanly
- [ ] Add compile-fail tests for feature-gated paths where API changes land
