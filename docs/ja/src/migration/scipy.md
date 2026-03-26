# SciPy/NumPyからの移行

このガイドでは、PythonのSciPy/NumPyからSciRS2への移行方法を説明します。
SciRS2はSciPyのAPIを可能な限り踏襲しているため、移行は比較的容易です。

## 基本的な違い

### 配列の扱い

NumPyの `ndarray` は SciRS2では `scirs2_core::ndarray` クレート（Rust ndarray）に対応します。

```python
# Python (NumPy)
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([5.0, 6.0])
```

```rust
// Rust (SciRS2)
use scirs2_core::ndarray::array;
let a = array![[1.0, 2.0], [3.0, 4.0]];
let b = array![5.0, 6.0];
```

### エラーハンドリング

Pythonでは例外が発生しますが、SciRS2では `Result` 型を使用します。

```python
# Python -- 例外が発生する可能性がある
result = scipy.linalg.solve(a, b)
```

```rust
// Rust -- 明示的なエラーハンドリング
let result = scirs2_linalg::solve(&a.view(), &b.view())?;
// '?' 演算子でエラーを呼び出し元に伝播
```

### ビュー（View）

SciRS2では多くの関数が配列のビュー（`ArrayView`）を受け取ります。
所有権の移動を防ぎ、不要なコピーを避けるためです。

```python
# Python -- 暗黙のビュー
scipy.linalg.det(a)
```

```rust
// Rust -- 明示的なビュー
scirs2_linalg::det(&a.view(), None)?;
```

## モジュール対応表

### scipy.linalg → scirs2-linalg

| Python | Rust | 備考 |
|--------|------|------|
| `scipy.linalg.solve(a, b)` | `solve(&a.view(), &b.view())?` | |
| `scipy.linalg.inv(a)` | `inv(&a.view(), None)?` | |
| `scipy.linalg.det(a)` | `det(&a.view(), None)?` | |
| `scipy.linalg.eigh(a)` | `eigh(&a.view())?` | 対称行列のみ |
| `scipy.linalg.svd(a)` | `svd(&a.view(), true)?` | |
| `scipy.linalg.lu(a)` | `lu(&a.view())?` | |
| `scipy.linalg.cholesky(a)` | `cholesky(&a.view())?` | 正定値のみ |

### scipy.stats → scirs2-stats

| Python | Rust | 備考 |
|--------|------|------|
| `norm(loc=0, scale=1)` | `Normal::new(0.0, 1.0)?` | |
| `dist.pdf(x)` | `dist.pdf(x)` | |
| `dist.cdf(x)` | `dist.cdf(x)?` | Result型 |
| `dist.ppf(q)` | `dist.ppf(q)?` | Result型 |
| `ttest_1samp(data, 0)` | `ttest_1samp(&data.view(), 0.0)?` | |

### scipy.signal → scirs2-signal

| Python | Rust | 備考 |
|--------|------|------|
| `butter(4, 0.2)` | `butter(4, &[0.2], None, None, None)?` | |
| `filtfilt(b, a, x)` | `filtfilt(&b, &a, &signal)?` | |
| `welch(x, fs=1000)` | `welch(&signal, Some(1000.0), ...)?` | |

### scipy.optimize → scirs2-optimize

| Python | Rust | 備考 |
|--------|------|------|
| `minimize(f, x0)` | `minimize(f, &x0, None, None)?` | |
| `brentq(f, a, b)` | `brentq(f, a, b, None)?` | |

## 型の対応

| Python/NumPy | Rust/SciRS2 |
|-------------|-------------|
| `float` / `np.float64` | `f64` |
| `complex` / `np.complex128` | `Complex64` |
| `np.ndarray` (1D) | `Array1<f64>` |
| `np.ndarray` (2D) | `Array2<f64>` |
| `np.ndarray` (view) | `ArrayView1<f64>` / `ArrayView2<f64>` |
| `None` | `Option<T>` |
| 例外 | `Result<T, E>` |

## 移行のヒント

1. **`unwrap()` を使わない**: SciRS2では `?` 演算子を使用してエラーを伝播してください。
2. **ビューを渡す**: 配列を関数に渡す際は `.view()` を使用してください。
3. **型を明示する**: Rustの型推論を活用しつつ、必要に応じて型注釈を追加してください。
4. **フィーチャーフラグを確認する**: 必要な機能がフィーチャーフラグの背後にある場合があります。
5. **scirs2-core経由でインポート**: `ndarray` を直接依存に追加せず、
   `scirs2_core::ndarray` から使用してください。
