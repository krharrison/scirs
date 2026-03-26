# Linear Algebra (scirs2-linalg)

`scirs2-linalg` provides comprehensive linear algebra operations with APIs that mirror
`scipy.linalg` and `numpy.linalg`. It covers basic operations, matrix decompositions,
eigenvalue problems, iterative solvers, matrix functions, and attention mechanisms.

## SciPy Equivalence Table

| SciPy / NumPy | SciRS2 | Description |
|----------------|--------|-------------|
| `numpy.linalg.det` | `det(&a.view(), None)` | Determinant |
| `numpy.linalg.inv` | `inv(&a.view(), None)` | Matrix inverse |
| `scipy.linalg.solve` | `solve(&a.view(), &b.view(), None)` | Solve Ax = b |
| `scipy.linalg.lu` | `lu(&a.view(), None)` | LU decomposition |
| `scipy.linalg.qr` | `qr(&a.view(), None)` | QR decomposition |
| `scipy.linalg.svd` | `svd(&a.view(), true, None)` | Singular value decomposition |
| `scipy.linalg.cholesky` | `cholesky(&a.view(), None)` | Cholesky decomposition |
| `scipy.linalg.eig` | `eig(&a.view(), None)` | Eigenvalues and eigenvectors |
| `scipy.linalg.expm` | `expm(&a.view())` | Matrix exponential |
| `numpy.linalg.norm` | `norm(&a.view(), order)` | Matrix / vector norms |

## Basic Operations

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::{det, inv, solve};

fn basics() -> Result<(), Box<dyn std::error::Error>> {
    let a = array![[4.0, 2.0], [2.0, 3.0]];

    // Determinant
    let d = det(&a.view(), None)?;
    assert!((d - 8.0).abs() < 1e-10);

    // Inverse
    let a_inv = inv(&a.view(), None)?;

    // Solve Ax = b
    let b = array![6.0, 7.0];
    let x = solve(&a.view(), &b.view(), None)?;

    Ok(())
}
```

## Matrix Decompositions

### LU Decomposition

Returns `(P, L, U)` such that `PA = LU`:

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::lu;

let a = array![[2.0, 1.0], [4.0, 3.0]];
let (p, l, u) = lu(&a.view(), None).expect("LU failed");
// L is unit lower triangular, U is upper triangular
```

### QR Decomposition

Returns `(Q, R)` such that `A = QR`:

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::qr;

let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let (q, r) = qr(&a.view(), None).expect("QR failed");
// Q is orthogonal (m x m), R is upper triangular (m x n)
```

### SVD

Returns `(U, S, Vt)` such that `A = U * diag(S) * Vt`:

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::svd;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let (u, s, vt) = svd(&a.view(), true, None).expect("SVD failed");
// s contains singular values in descending order
```

### Cholesky Decomposition

For symmetric positive-definite matrices, returns lower triangular `L` such that `A = L * L^T`:

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::cholesky;

let spd = array![[4.0, 2.0], [2.0, 3.0]];
let l = cholesky(&spd.view(), None).expect("Cholesky failed");
```

### Eigenvalue Decomposition

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::eig;

let a = array![[1.0, 2.0], [3.0, 4.0]];
let (eigenvalues, eigenvectors) = eig(&a.view(), None).expect("Eig failed");
// eigenvalues is Array1<Complex64>, eigenvectors is Array2<Complex64>
```

## Iterative Solvers

For large or sparse systems where direct decomposition is impractical, use iterative methods:

```rust,ignore
use scirs2_linalg::iterative::{cg, gmres, bicgstab, IterativeOptions};

// Conjugate Gradient (symmetric positive-definite systems)
let options = IterativeOptions::default()
    .with_tolerance(1e-10)
    .with_max_iterations(1000);
let x = cg(&a, &b, &options)?;

// GMRES (general non-symmetric systems)
let x = gmres(&a, &b, &options)?;

// BiCGSTAB (non-symmetric, often faster than GMRES)
let x = bicgstab(&a, &b, &options)?;
```

## Matrix Functions

Compute functions of matrices (not element-wise -- these are true matrix functions):

```rust,ignore
use scirs2_linalg::{expm, logm, sqrtm};

// Matrix exponential: exp(A)
let exp_a = expm(&a.view())?;

// Matrix logarithm: log(A)
let log_a = logm(&a.view())?;

// Matrix square root: A^(1/2)
let sqrt_a = sqrtm(&a.view())?;
```

## Advanced Features

### Mixed-Precision GEMM

`scirs2-linalg` supports f16 and bf16 matrix multiplication for memory-constrained
workloads with automatic accumulation in higher precision:

```rust,ignore
use scirs2_linalg::mixed_precision::{f16_gemm, bf16_gemm};

// f16 inputs, f32 accumulation, f16 output
let c = f16_gemm(&a_f16, &b_f16)?;
```

### Distributed Linear Algebra

SUMMA-based distributed matrix multiplication and communication-avoiding QR for
multi-node deployments:

```rust,ignore
use scirs2_linalg::distributed::{DistributedMatrix, summa_multiply};

let dist_a = DistributedMatrix::new(local_block, grid_position, grid_dims);
let dist_c = summa_multiply(&dist_a, &dist_b)?;
```

### Block Krylov Methods

For computing multiple eigenvalues/singular values simultaneously:

```rust,ignore
use scirs2_linalg::block_krylov::BlockLanczos;

let solver = BlockLanczos::new(block_size);
let (eigenvalues, eigenvectors) = solver.solve(&a, num_eigenvalues)?;
```

### Hierarchical Matrices (H2)

For N-body problems and boundary element methods where the full matrix has
off-diagonal low-rank structure:

```rust,ignore
use scirs2_linalg::hmatrix_h2::H2Matrix;

let h2 = H2Matrix::from_kernel(points, kernel_fn, tolerance)?;
let y = h2.matvec(&x)?;  // O(n) instead of O(n^2)
```
