# プロジェクト構成

SciRS2はCargoワークスペースとして構成されており、約29のクレートで構成されています。
各クレートは独立して利用でき、必要な機能だけを選択的に取り込めます。

## ディレクトリ構造

```
scirs/
├── Cargo.toml          # ワークスペースルート
├── scirs2-core/        # 基盤クレート（ndarray再エクスポート、共有型）
├── scirs2-linalg/      # 線形代数
├── scirs2-stats/       # 統計
├── scirs2-signal/      # 信号処理
├── scirs2-fft/         # 高速フーリエ変換
├── scirs2-optimize/    # 最適化
├── scirs2-integrate/   # 数値積分・微分方程式
├── scirs2-interpolate/ # 補間
├── scirs2-special/     # 特殊関数
├── scirs2-sparse/      # 疎行列
├── scirs2-ndimage/     # 画像処理
├── scirs2-neural/      # ニューラルネットワーク
├── scirs2-graph/       # グラフニューラルネットワーク
├── scirs2-series/      # 時系列解析
├── scirs2-text/        # 自然言語処理
├── scirs2-vision/      # コンピュータビジョン
├── scirs2-io/          # データI/O
├── scirs2-datasets/    # データセット
├── scirs2-metrics/     # メトリクス
├── scirs2-cluster/     # クラスタリング
├── scirs2-transform/   # データ変換
├── scirs2-wasm/        # WebAssemblyバインディング
└── docs/               # ドキュメント（本書）
```

## クレート依存関係

すべてのクレートは `scirs2-core` に依存しています。
`scirs2-core` は `ndarray` と `num-complex` を再エクスポートし、
共通のエラー型やユーティリティ関数を提供します。

```
scirs2-core  ← すべてのクレートの基盤
├── scirs2-linalg
│   ├── scirs2-sparse （疎行列の線形代数）
│   └── scirs2-integrate （線形代数を使用する数値計算）
├── scirs2-stats
│   └── scirs2-metrics （統計的メトリクス）
├── scirs2-fft
│   └── scirs2-signal （FFTベースの信号処理）
└── scirs2-neural
    └── scirs2-graph （グラフニューラルネットワーク）
```

## バージョン管理

すべてのクレートは同一のバージョン番号を共有しています。
ワークスペースの `Cargo.toml` で一元管理されます。

```toml
# ルートCargo.toml
[workspace.package]
version = "0.4.0"
```

各クレートの `Cargo.toml` では `version.workspace = true` を使用します。

## Pure Rustポリシー

SciRS2のデフォルトビルドはC/Fortranへの依存を一切含みません。
GPU機能などの外部ランタイム依存はフィーチャーフラグで隔離されています。

| Pure Rustライブラリ | 置換対象 |
|-------------------|---------|
| OxiBLAS | OpenBLAS / MKL |
| OxiFFT | FFTW / RustFFT |
| OxiARC | zlib / zstd / flate2 |
| OxiCode | bincode |

この設計により、`cargo build` だけでクロスコンパイルを含むすべてのプラットフォームで
ビルドが完結します。
