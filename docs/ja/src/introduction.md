# SciRS2 -- Rust科学計算ライブラリ

SciRS2は、Rustで実装された包括的な科学計算ライブラリです。
PythonのSciPy/NumPyに相当する機能を、Rustの型安全性・メモリ安全性・高パフォーマンスと共に提供します。

## 設計思想

SciRS2は以下の原則に基づいて設計されています。

- **Pure Rust**: C/Fortranコンパイラを一切必要としません。BLAS演算にはOxiBLAS、FFTにはOxiFFT、
  圧縮処理にはOxiARCを使用しています。クロスコンパイルが容易で、システム依存性の問題を排除しています。

- **SciPy API互換**: 関数名、引数の順序、戻り値の型をSciPyに可能な限り合わせています。
  `scipy.linalg.solve` を知っていれば、`scirs2_linalg::solve` の使い方もわかります。

- **型安全性**: エラーは `Result` 型で返され、パニックは使用しません。
  ワークスペース全体で `unwrap()` の使用を禁止するポリシーを適用しており、
  本番コードではすべてのエラーパスを明示的に処理します。

- **デフォルトで高性能**: SIMD最適化、Rayonによる並列処理、キャッシュフレンドリーな
  データレイアウトを全体にわたって使用しています。オプションのGPUアクセラレーション
  （CUDA、ROCm、Metal）はフィーチャーフラグで有効にできます。

## 機能概要

SciRS2は約29のワークスペースクレートで構成され、以下の分野をカバーしています。

| 分野 | クレート | SciPy/Python相当 |
|------|---------|-----------------|
| 線形代数 | `scirs2-linalg` | `scipy.linalg`, `numpy.linalg` |
| 統計 | `scirs2-stats` | `scipy.stats` |
| 信号処理 | `scirs2-signal` | `scipy.signal` |
| フーリエ変換 | `scirs2-fft` | `scipy.fft` |
| 最適化 | `scirs2-optimize` | `scipy.optimize` |
| 積分/常微分方程式/偏微分方程式 | `scirs2-integrate` | `scipy.integrate` |
| 補間 | `scirs2-interpolate` | `scipy.interpolate` |
| 特殊関数 | `scirs2-special` | `scipy.special` |
| 疎行列 | `scirs2-sparse` | `scipy.sparse` |
| 画像処理 | `scirs2-ndimage` | `scipy.ndimage` |
| ニューラルネットワーク | `scirs2-neural` | PyTorch / TensorFlow |
| グラフニューラルネットワーク | `scirs2-graph` | PyG, DGL |
| 時系列解析 | `scirs2-series` | statsmodels, Darts |
| 自然言語処理 | `scirs2-text` | HuggingFace, gensim |
| コンピュータビジョン | `scirs2-vision` | torchvision, OpenCV |
| データI/O | `scirs2-io` | pandas, pyarrow |
| データセット | `scirs2-datasets` | sklearn.datasets, HF datasets |
| メトリクス | `scirs2-metrics` | sklearn.metrics, torchmetrics |
| クラスタリング | `scirs2-cluster` | sklearn.cluster |
| データ変換 | `scirs2-transform` | sklearn.preprocessing |
| WebAssembly | `scirs2-wasm` | -- |

## SciPyとの比較

SciRS2はSciPyとほぼ同等のアルゴリズム範囲をカバーし、さらに以下の拡張を提供します。

- **ニューラルネットワーク層と学習機能**: アテンション機構、量子化（GPTQ, AWQ, SmoothQuant）、
  NAS（DARTS, GDAS, SNAS）、分散学習（パイプライン/テンソル並列）。
- **グラフニューラルネットワーク**: GCN, GAT, GraphSAGE, R-GCN, HGT, GraphGPS, Graphormer、
  符号付き/有向グラフ埋め込み。
- **高度なPDEソルバー**: 不連続ガレルキン法、仮想要素法、ペリダイナミクス、
  物理情報ニューラルネットワーク（PINN）。
- **WebAssemblyサポート**: `scirs2-wasm` を使用して、WebGPUアクセラレーションと共に
  FFT、線形代数、信号処理をブラウザ上で実行可能。

SciPyの強みはエコシステムの成熟度とサードパーティ統合の幅広さにあります。
SciRS2はコンパイル時安全性、ゼロコスト抽象化、そしてサーバー・組み込みデバイス・
ブラウザを問わず同一コードをデプロイできる利点で補っています。

## COOLJAPANエコシステム

SciRS2はCOOLJAPANオープンソースエコシステムの一部です。
COOLJAPANは、C/Fortran科学計算ライブラリのPure Rust代替を提供しています。

| ライブラリ | 置換対象 |
|-----------|---------|
| OxiBLAS | OpenBLAS, MKL |
| OxiFFT | FFTW |
| OxiARC | zlib, zstd, bzip2 |
| OxiCode | bincode |
| OxiZ | Z3 SMTソルバー |

すべてのCOOLJAPANライブラリは、Pure Rustコンパイル、可能な限りunsafeを排除、
幅広いプラットフォームサポート（Linux, macOS, Windows, WASM, iOS, Android）を共有しています。

## サポート

- [GitHub Issues](https://github.com/cool-japan/scirs/issues) -- バグ報告と機能リクエスト
- [APIドキュメント](https://docs.rs/scirs2) -- 自動生成されたrustdoc
- 本書 -- チュートリアル、ガイド、移行リファレンス
