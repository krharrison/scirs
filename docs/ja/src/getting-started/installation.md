# インストール

## 必要要件

- Rust 1.75以降（2021 edition）
- C/Fortranコンパイラは不要 -- SciRS2はデフォルトでPure Rustです

## プロジェクトへのSciRS2の追加

最も簡単な方法は、すべてのモジュールを再エクスポートするアンブレラクレートに依存することです。

```toml
[dependencies]
scirs2 = "0.4.0"
```

より細かい制御が必要な場合は、個別のクレートに依存できます。

```toml
[dependencies]
scirs2-core = "0.4.0"
scirs2-linalg = "0.4.0"
scirs2-stats = "0.4.0"
scirs2-fft = "0.4.0"
```

すべてのクレートは `version.workspace = true` メカニズムにより同じバージョン番号を共有しています。

## フィーチャーフラグ

ほとんどのクレートは拡張機能のためのオプションのフィーチャーフラグを提供しています。

| フィーチャー | 説明 | 利用可能なクレート |
|------------|------|-----------------|
| `simd` | SIMD高速化演算（AVX/AVX2/AVX-512） | linalg, signal, fft |
| `parallel` | Rayonによるマルチスレッド実行 | linalg, fft, sparse |
| `gpu` | GPUアクセラレーション（CUDA/ROCm/Metal/OpenCL） | linalg, fft, sparse, neural |
| `serde` | シリアライゼーションサポート | core, stats, sparse |

フィーチャーを有効にする例:

```toml
[dependencies]
scirs2-linalg = { version = "0.4.0", features = ["simd", "parallel"] }
scirs2-fft = { version = "0.4.0", features = ["parallel"] }
```

## コアクレート

`scirs2-core` はすべてのSciRS2クレートが依存する基盤クレートです。
`ndarray` と `num-complex` を再エクスポートするため、別途追加する必要はありません。

```rust
use scirs2_core::ndarray::{array, Array1, Array2, ArrayView1};
use scirs2_core::numeric::Complex64;
```

これによりワークスペース全体でバージョンの一貫性が保たれます。
`ndarray` の型は直接依存を追加するのではなく、常に `scirs2_core` 経由でインポートしてください。

## インストールの確認

小さなテストプログラムを作成します。

```rust
use scirs2_core::ndarray::array;
use scirs2_linalg::{det, inv};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2x2行列を作成
    let a = array![[4.0, 2.0], [2.0, 3.0]];

    // 行列式を計算
    let d = det(&a.view(), None)?;
    println!("行列式: {}", d);

    // 逆行列を計算
    let a_inv = inv(&a.view(), None)?;
    println!("逆行列:\n{}", a_inv);

    Ok(())
}
```

`cargo run` で実行してください。行列式（8.0）と逆行列が表示されれば、正しくインストールされています。

## プラットフォームサポート

SciRS2はRustがサポートするすべてのプラットフォームでコンパイルできます。

| プラットフォーム | 状態 | 備考 |
|----------------|------|------|
| Linux (x86_64, aarch64) | 完全サポート | CI テスト済み |
| macOS (x86_64, Apple Silicon) | 完全サポート | CI テスト済み |
| Windows (x86_64) | 完全サポート | CI テスト済み |
| WebAssembly (wasm32) | `scirs2-wasm` 経由 | ブラウザおよびNode.js |
| iOS (aarch64) | コアモジュール | 最小iOS 13.0 |
| Android (aarch64, armv7) | コアモジュール | 最小API 21 |

GPU機能は対応するランタイム（CUDAツールキット、ROCmなど）が必要ですが、
完全にオプションでありコンパイルには不要です。
