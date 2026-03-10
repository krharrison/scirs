# scirs2-sparse Development TODO

## v0.3.1 — COMPLETED

### New Sparse Formats
- ELLPACK/ITPACK format with uniform nnz-per-row storage (GPU-friendly)
- BCSR (Block CSR) for problems with dense block substructure
- `SymCsrArray` / `SymCooArray` — symmetric half-storage variants
- Enhanced index dtype handling with automatic i32/i64 selection

### Eigenvalue Solvers
- LOBPCG (Locally Optimal Block Preconditioned CG) for extreme eigenvalues
- IRAM (Implicitly Restarted Arnoldi Method) for non-symmetric matrices
- Shift-and-invert eigenvalue mode for interior spectrum
- Generalized eigenvalue problem `Ax = λBx`
- Lanczos iteration for symmetric SPD matrices

### Advanced Preconditioners
- Block Jacobi preconditioner with block-diagonal factorization
- SPAI (Sparse Approximate Inverse) via minimization in Frobenius norm
- Additive Schwarz method with configurable overlap
- Restricted Additive Schwarz (RAS)
- Two-level Schwarz with coarse correction
- Smoothed aggregation AMG (Algebraic Multigrid) — full setup and solve cycle

### Iterative Solvers
- SYMMLQ for symmetric indefinite systems
- LGMRES (augmented GMRES with deflation)
- Recycled Krylov (GCRO-DR style)
- LSQR and LSMR for least-squares problems
- SOR and SSOR iterative relaxation solvers
- Saddle-point block preconditioned solver
- GCROT and TFQMR

### Graph Algorithms (csgraph)
- Graph Laplacian: standard, normalized, random-walk variants
- Algebraic connectivity (Fiedler eigenvalue/vector) via LOBPCG
- Spectral clustering using sparse eigensolver output
- Enhanced connected component labeling with weak/strong modes

### Hierarchical Matrices
- H-matrix structure (row/column cluster tree)
- Adaptive cross approximation (ACA) for off-diagonal block compression
- H-matrix-vector multiply
- H-matrix preconditioner apply

### Domain Decomposition
- Additive Schwarz with overlap
- Restricted Additive Schwarz
- Two-level method with coarse-grid solve
- Subdomain interface identification from CSR graph

### Neural Adaptive Sparse
- Neural network for sparsity pattern prediction
- Learned sparse preconditioner training loop
- Data-driven fill-reducing reordering heuristics

### Other Additions
- Sparse matrix exponential (`expm` via scaling/squaring on CSR)
- `expm_multiply`: compute `expm(A) @ v` without forming `expm(A)`
- Saddle-point system (block 2x2) specialized solver
- Smoothed aggregation setup: strength of connection, aggregation, prolongation smoothing
- Sparse format benchmark suite

---

## v0.4.0 — Planned

### GPU Sparse BLAS
- [ ] CSR SpMV on GPU via compute shaders (OxiFFT GPU backend model)
- [ ] GPU-accelerated COO/CSR construction from triplets
- [ ] GPU BiCGSTAB and CG solvers
- [ ] Mixed CPU/GPU preconditioning (ILU on CPU, SpMV on GPU)

### Distributed Sparse Solvers
- [ ] Distributed CSR with row-based partitioning
- [ ] Distributed SpMV with halo exchange
- [ ] Distributed AMG via `scirs2-core` ring allreduce

### Graph Algorithm Enhancements
- [ ] Approximate graph coloring for parallel Gauss-Seidel
- [ ] Nested dissection reordering (via graph partitioning)
- [ ] Cuthill-McKee and reverse Cuthill-McKee reordering
- [ ] Approximate minimum degree (AMD) reordering

### Format Enhancements
- [ ] Sliced ELLPACK (SELL) for variable nnz-per-row with GPU padding
- [ ] CSR5 format for GPU-friendly SpMV
- [ ] Compressed sparse fiber (CSF) for sparse tensors

### Additional Preconditioners
- [ ] Multilevel ILU (MILUE) with coarse correction
- [ ] Sparse approximate inverse via AINV (robust incomplete factorization)
- [ ] Polynomial preconditioner (Chebyshev acceleration)

---

## Known Issues / Technical Debt

- `krylov.rs` was deleted and replaced by the `krylov/` submodule; ensure no lingering re-exports break downstream
- `neural_adaptive_sparse/neural_network.rs` needs more training data and documented hyperparameters
- H-matrix implementation is a structural sketch; ACA convergence not yet validated against dense reference
- GPU sparse stubs in `linalg/` are feature-gated but untested without actual GPU; add mock tests
- Several solver files exceed 2000 lines; use `rslines 50` to identify split candidates
- SPAI preconditioner setup cost is O(n * bandwidth^2); document when to prefer it over ILU
- Saddle-point solver assumes 2x2 block structure; generalize to block-n
