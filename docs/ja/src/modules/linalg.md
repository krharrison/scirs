# 線形代数 (scirs2-linalg)

`scirs2-linalg` はSciRS2の線形代数モジュールです。
SciPyの `scipy.linalg` および `numpy.linalg` に相当する機能を提供します。

## 主な機能

- 連立一次方程式の求解
- 行列分解（LU, QR, SVD, Cholesky, 固有値分解）
- 行列関数（行列指数関数、行列対数関数）
- 構造化行列（Toeplitz, Circulant, Hankel）
- 分散線形代数（SUMMA, CAQR）
- 混合精度演算（f16/bf16 GEMM）

## 基本的な操作

### 連立一次方程式

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::solve;

// Ax = b を解く
let a = array![[3.0, 1.0], [1.0, 2.0]];
let b = array![9.0, 8.0];
let x = solve(&a.view(), &b.view())?;
// x = [2.0, 3.0]
```

### LU分解

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::lu;

let a = array![[2.0, 1.0], [4.0, 3.0]];
let (p, l, u) = lu(&a.view())?;
// PA = LU
// L: 下三角行列、U: 上三角行列、P: 置換行列
```

### 特異値分解 (SVD)

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::svd;

let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let (u, s, vt) = svd(&a.view(), true)?;
// A = U * diag(S) * V^T
// s: 特異値（降順）
```

### 固有値分解

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::eigh;

// 対称行列の固有値分解
let a = array![[2.0, 1.0], [1.0, 3.0]];
let (eigenvalues, eigenvectors) = eigh(&a.view())?;
// eigenvalues: 固有値（昇順）
// eigenvectors: 固有ベクトル（列ベクトル）
```

### Cholesky分解

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::cholesky;

// 正定値対称行列のCholesky分解
let a = array![[4.0, 2.0], [2.0, 3.0]];
let l = cholesky(&a.view())?;
// A = L * L^T
```

## 高度な機能

### 行列関数

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::matrix_functions::{expm, logm};

let a = array![[1.0, 0.5], [0.0, 2.0]];

// 行列指数関数 exp(A)
let exp_a = expm(&a.view())?;

// 行列対数関数 log(A)
let log_a = logm(&a.view())?;
```

### 構造化行列

```rust
use scirs2_linalg::structured::{toeplitz_solve, circulant_multiply};

// Toeplitz行列の高速求解（O(n log n)）
let first_col = vec![4.0, 3.0, 2.0, 1.0];
let first_row = vec![4.0, 5.0, 6.0, 7.0];
let b = vec![1.0, 2.0, 3.0, 4.0];
let x = toeplitz_solve(&first_col, &first_row, &b)?;
```

## SciPy対応表

| SciPy | SciRS2 | 説明 |
|-------|--------|------|
| `scipy.linalg.solve` | `scirs2_linalg::solve` | 連立一次方程式 |
| `scipy.linalg.inv` | `scirs2_linalg::inv` | 逆行列 |
| `scipy.linalg.det` | `scirs2_linalg::det` | 行列式 |
| `scipy.linalg.lu` | `scirs2_linalg::lu` | LU分解 |
| `scipy.linalg.qr` | `scirs2_linalg::qr` | QR分解 |
| `scipy.linalg.svd` | `scirs2_linalg::svd` | 特異値分解 |
| `scipy.linalg.eigh` | `scirs2_linalg::eigh` | 対称行列の固有値分解 |
| `scipy.linalg.cholesky` | `scirs2_linalg::cholesky` | Cholesky分解 |
| `scipy.linalg.expm` | `scirs2_linalg::matrix_functions::expm` | 行列指数関数 |
| `numpy.linalg.norm` | `scirs2_linalg::norm` | 行列/ベクトルノルム |
