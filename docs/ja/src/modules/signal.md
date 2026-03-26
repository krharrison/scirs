# 信号処理 (scirs2-signal)

`scirs2-signal` はSciRS2の信号処理モジュールです。
SciPyの `scipy.signal` に相当する機能を提供します。

## 主な機能

- フィルタ設計（バターワース、チェビシェフ、楕円）
- フィルタリング（FIR, IIR, ゼロ位相フィルタリング）
- 窓関数（Hann, Hamming, Blackman, Kaiser）
- スペクトル解析（ウェルチ法、スペクトログラム）
- ビームフォーミング（MVDR, STAP, サブアレイ）
- モーダル解析
- リアルタイムパイプライン

## フィルタ設計

### バターワースフィルタ

```rust
use scirs2_signal::filter_design::butter;

// 4次バターワースローパスフィルタ
// カットオフ周波数: 100 Hz / (サンプリング周波数/2)
let (b, a) = butter(4, &[0.2], None, None, None)?;
// b: 分子係数、a: 分母係数

// バンドパスフィルタ
let (b, a) = butter(
    3,                    // フィルタ次数
    &[0.1, 0.4],          // 正規化カットオフ周波数 [low, high]
    Some("bandpass"),      // フィルタタイプ
    None,
    None,
)?;
```

### チェビシェフフィルタ

```rust
use scirs2_signal::filter_design::cheby1;

// タイプIチェビシェフフィルタ（リプル1dB）
let (b, a) = cheby1(4, 1.0, &[0.2], None, None, None)?;
```

## フィルタリング

```rust
use scirs2_signal::filtering::{lfilter, filtfilt, sosfilt};

// 因果フィルタリング（遅延あり）
let filtered = lfilter(&b, &a, &signal)?;

// ゼロ位相フィルタリング（遅延なし、位相歪みなし）
let filtered = filtfilt(&b, &a, &signal)?;
```

## 窓関数

```rust
use scirs2_signal::windows::{hann, hamming, blackman, kaiser};

let n = 256;

// Hann窓
let w = hann(n, true)?;

// Kaiser窓（beta=8.6 で約90dBのサイドローブ減衰）
let w = kaiser(n, 8.6, true)?;
```

## スペクトル解析

### ウェルチ法

```rust
use scirs2_signal::spectral::welch;

// パワースペクトル密度推定
let (freqs, psd) = welch(
    &signal,
    Some(1000.0),  // サンプリング周波数
    None,          // 窓関数（デフォルト: Hann）
    Some(256),     // セグメント長
    Some(128),     // オーバーラップ
    None,          // FFT長
)?;
```

### スペクトログラム

```rust
use scirs2_signal::spectral::spectrogram;

// 短時間フーリエ変換ベースのスペクトログラム
let (t, f, sxx) = spectrogram(
    &signal,
    Some(1000.0),  // サンプリング周波数
    None,          // 窓関数
    Some(256),     // セグメント長
    Some(128),     // オーバーラップ
)?;
// t: 時間軸、f: 周波数軸、sxx: パワースペクトル密度
```

## SciPy対応表

| SciPy | SciRS2 | 説明 |
|-------|--------|------|
| `scipy.signal.butter` | `filter_design::butter` | バターワースフィルタ設計 |
| `scipy.signal.cheby1` | `filter_design::cheby1` | チェビシェフI型フィルタ |
| `scipy.signal.filtfilt` | `filtering::filtfilt` | ゼロ位相フィルタリング |
| `scipy.signal.welch` | `spectral::welch` | PSD推定 |
| `scipy.signal.spectrogram` | `spectral::spectrogram` | スペクトログラム |
| `scipy.signal.windows.*` | `windows::*` | 窓関数 |
