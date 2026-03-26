# Sparse Matrices (scirs2-sparse)

`scirs2-sparse` provides sparse matrix formats and operations modeled after `scipy.sparse`,
supporting CSR, CSC, COO, DOK, LIL, DIA, and BSR formats with efficient linear algebra,
solvers, and preconditioners.

## Matrix Formats

### CSR (Compressed Sparse Row)

The default format for row-oriented operations and sparse matrix-vector multiplication:

```rust
use scirs2_sparse::csr_array::CsrArray;

fn csr_demo() -> Result<(), Box<dyn std::error::Error>> {
    // Create from triplets (row, col, value)
    let rows = vec![0, 0, 1, 2, 2];
    let cols = vec![0, 2, 2, 0, 1];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sparse = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false)?;

    println!("Shape: {:?}", sparse.shape());
    println!("NNZ: {}", sparse.nnz());
    Ok(())
}
```

### CSC (Compressed Sparse Column)

Efficient for column slicing and transpose operations:

```rust,ignore
use scirs2_sparse::csc_array::CscArray;

let sparse = CscArray::from_triplets(&rows, &cols, &data, (3, 3), false)?;
```

### COO (Coordinate Format)

Best for incremental construction and format conversion:

```rust,ignore
use scirs2_sparse::coo_array::CooArray;

let coo = CooArray::from_triplets(&rows, &cols, &data, (3, 3))?;

// Convert to CSR for efficient arithmetic
let csr: CsrArray<f64> = coo.to_csr()?;
```

### DOK (Dictionary of Keys)

Best for random access and incremental construction:

```rust,ignore
use scirs2_sparse::dok_array::DokArray;

let mut dok = DokArray::new((1000, 1000));
dok.set(0, 5, 3.14)?;
dok.set(42, 99, 2.71)?;

// Convert to CSR when done building
let csr = dok.to_csr()?;
```

## Sparse Arithmetic

```rust,ignore
use scirs2_sparse::csr_array::CsrArray;

// Matrix-vector multiplication
let y = sparse.dot(&x)?;

// Matrix-matrix multiplication
let c = a.dot_matrix(&b)?;

// Scalar operations
let scaled = sparse.scale(2.0)?;

// Transpose
let at = sparse.transpose()?;
```

## Sparse Linear Solvers

### Direct Solvers

```rust,ignore
use scirs2_sparse::linalg::{spsolve, splu};

// Direct solve: Ax = b
let x = spsolve(&a, &b)?;

// Sparse LU factorization (reusable for multiple right-hand sides)
let lu = splu(&a)?;
let x1 = lu.solve(&b1)?;
let x2 = lu.solve(&b2)?;
```

### Iterative Solvers

```rust,ignore
use scirs2_sparse::linalg::{cg, gmres, bicgstab};

// Conjugate Gradient (symmetric positive-definite)
let (x, info) = cg(&a, &b, None, Some(1e-10), Some(1000))?;

// GMRES (general)
let (x, info) = gmres(&a, &b, None, Some(1e-10), Some(1000), Some(50))?;

// BiCGSTAB
let (x, info) = bicgstab(&a, &b, None, Some(1e-10), Some(1000))?;
```

## Preconditioners

### Incomplete LU / Cholesky

```rust,ignore
use scirs2_sparse::preconditioners::{ilu, ichol};

// ILU(0) preconditioner
let precond = ilu(&a, 0)?;

// Incomplete Cholesky (for SPD matrices)
let precond = ichol(&a)?;

// Use with iterative solver
let (x, info) = cg(&a, &b, Some(&precond), Some(1e-10), Some(1000))?;
```

### Algebraic Multigrid (AMG)

```rust,ignore
use scirs2_sparse::amg::{AMGSolver, AMGOptions};

let options = AMGOptions::default()
    .with_max_levels(10)
    .with_coarsening("ruge_stuben");

let amg = AMGSolver::new(&a, options)?;
let x = amg.solve(&b, 1e-10, 100)?;
```

### Advanced Preconditioners

```rust,ignore
use scirs2_sparse::preconditioners::{ainv, milue};

// Approximate Inverse preconditioner
let precond = ainv(&a, drop_tolerance)?;

// MILUE (Modified ILU with Enhanced stability)
let precond = milue(&a, fill_level)?;
```

## Eigenvalue Problems

```rust,ignore
use scirs2_sparse::linalg::{eigs, eigsh};

// Largest k eigenvalues of a general sparse matrix
let (eigenvalues, eigenvectors) = eigs(&a, k, "LM")?;

// Largest k eigenvalues of a symmetric sparse matrix
let (eigenvalues, eigenvectors) = eigsh(&a, k, "LM")?;

// Smallest eigenvalues
let (eigenvalues, eigenvectors) = eigsh(&a, k, "SM")?;
```

## GPU Acceleration

```rust,ignore
use scirs2_sparse::gpu::{GpuSparseMatrix, gpu_spmv};

// Transfer to GPU
let gpu_a = GpuSparseMatrix::from_csr(&a)?;
let gpu_x = gpu_a.upload_vector(&x)?;

// GPU SpMV
let gpu_y = gpu_spmv(&gpu_a, &gpu_x)?;
let y = gpu_y.download()?;
```

## SciPy Equivalence Table

| SciPy | SciRS2 |
|-------|--------|
| `scipy.sparse.csr_array` | `csr_array::CsrArray` |
| `scipy.sparse.csc_array` | `csc_array::CscArray` |
| `scipy.sparse.coo_array` | `coo_array::CooArray` |
| `scipy.sparse.linalg.spsolve` | `linalg::spsolve` |
| `scipy.sparse.linalg.cg` | `linalg::cg` |
| `scipy.sparse.linalg.gmres` | `linalg::gmres` |
| `scipy.sparse.linalg.eigs` | `linalg::eigs` |
| `scipy.sparse.linalg.eigsh` | `linalg::eigsh` |
