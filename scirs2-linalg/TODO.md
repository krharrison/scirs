# scirs2-linalg Development TODO

## v0.3.3 — COMPLETED

### Iterative Solvers
- GMRES (restarted) with configurable restart parameter
- GMRES-DR / recycled Krylov subspace (GCRO-DR style) — rewritten Feb 26, 2026
- Augmented Krylov (LGMRES-style) deflation
- Preconditioned Conjugate Gradient (PCG) with flexible preconditioning
- BiCGStab (stabilized bi-conjugate gradient)
- MINRES, SYMMLQ for symmetric indefinite systems
- Arnoldi iteration and thick-restart Lanczos
- Rewritten Lanczos QL eigensolver (fixed Feb 26, 2026)

### Randomized Linear Algebra
- Randomized SVD: Halko-Martinsson-Tropp with power iteration and oversampling
- Nystrom extension for low-rank PSD kernel approximation
- Randomized eigensolvers via subspace iteration
- Sketching: Gaussian sketch, CountSketch, SRHT (Subsampled Randomized Hadamard Transform)

### Tensor Decompositions
- CP-ALS (Canonical Polyadic via Alternating Least Squares)
- Tucker-HOOI (Higher-Order Orthogonal Iteration)
- Tensor contractions and mode-n products
- Hierarchical Tucker basics
- Tensor-train format representation

### Matrix Functions
- `expm` via Pade approximant with scaling and squaring
- `logm` via inverse scaling and squaring (Schur-based)
- `sqrtm` via Schur decomposition (Björck-Hammarling)
- `signm` via Newton iteration
- Matrix trigonometric functions (Schur-based): sin, cos, tan, sinh, cosh
- Polar decomposition (QDWH algorithm)
- Pade approximant module for arbitrary rational approximations

### Control Theory
- Continuous algebraic Riccati equation (CARE) via Newton + Hamiltonian Schur
- Discrete algebraic Riccati equation (DARE)
- Lyapunov equation (continuous and discrete, Bartels-Stewart)
- Sylvester equation solver (Bartels-Stewart with 2x2 block fix, Feb 26, 2026)
- Controllability / observability Gramians
- Balanced truncation model order reduction

### Structured Matrices
- Toeplitz matvec via FFT (O(n log n))
- Circulant diagonalization via FFT
- Hankel matvec
- Cauchy matrix: O(n^2) matvec, displacement structure
- Companion matrix and polynomial root finding
- Block tridiagonal direct solver

### Numerical Analysis
- Perturbation bounds for eigenvalues and singular values (Davis-Kahan, Weyl)
- Backward error analysis for linear systems
- Numerical range (field of values) estimation
- Condition number estimation (LAPACK-style power method)
- Matrix pencil solver for polynomial eigenvalue problems

### Other Additions
- CUR decomposition (DEIM-based column/row selection)
- Nuclear norm minimization (alternating projections, soft-impute)
- Matrix completion with soft-impute algorithm
- Indefinite LDL^T factorization (Bunch-Kaufman pivoting)
- Sparse-dense hybrid operations
- `number_theory.rs`: modular arithmetic, integer lattice algorithms

---

## v0.4.0 — Planned

### Mixed-Precision Arithmetic
- [ ] f16/bf16 storage with f32 accumulation for matmul
- [ ] Iterative refinement with higher-precision residual correction
- [ ] Automatically select precision based on condition number estimate

### Structured Matrix Exploits
- [ ] Hierarchical matrix (H-matrix) compression for dense-but-rank-structured matrices
- [ ] H^2-matrix arithmetic (O(n log n) matrix-vector products)
- [ ] Sequentially semi-separable (SSS) matrix operations

### Distributed Linear Algebra
- [ ] Distributed dense matmul (ScaLAPACK-style 2D block cyclic layout)
- [ ] Distributed QR via Householder with communication-avoiding variants
- [ ] Distributed SVD via Lanczos

### GPU Acceleration
- [ ] GPU-accelerated GEMM via OxiBLAS GPU backend
- [ ] GPU eigensolvers (cuSOLVER-equivalent in pure Rust)
- [ ] Mixed CPU/GPU solver: factor on GPU, refine on CPU

### Additional Algorithms
- [ ] Rank-revealing QR (RRQR) with column pivoting
- [ ] URV decomposition for rank-deficient systems
- [ ] Contour integral eigensolver (FEAST)
- [ ] Zolotarev rational approximations for matrix functions

---

## Known Issues / Technical Debt

- Some matrix function files exceed 2000 lines; use `rslines 50` to find candidates for splitting
- Lanczos eigensolver was rewritten Feb 26, 2026 after QL deflation bug; needs more stress tests on near-degenerate spectra
- Bartels-Stewart Sylvester 2x2 block handling was patched Feb 26, 2026; audit complex case
- GMRES recycled Krylov was substantially rewritten Feb 26, 2026; regression tests cover Poisson/convection-diffusion but not all corner cases
- Quantization-aware operations in `quantization/` need benchmarks comparing to GGML/bitsandbytes reference
- Control theory module (`control/`) lacks integration tests against MATLAB/Octave reference values
