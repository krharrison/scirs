# FFT (scirs2-fft)

`scirs2-fft` はSciRS2の高速フーリエ変換モジュールです。
SciPyの `scipy.fft` に相当する機能を提供します。
内部的にはOxiFFT（Pure Rust FFTライブラリ）を使用しています。

## 主な機能

- 1次元/多次元FFT/IFFT
- 実数FFT (rfft/irfft)
- 離散コサイン変換 (DCT)
- 離散正弦変換 (DST)
- 分数フーリエ変換 (FrFT)
- Wigner-Ville分布
- 適応的スパースFFT

## 基本的なFFT

```rust
use scirs2_core::ndarray::Array1;
use scirs2_fft::{fft, ifft, fftfreq};

// 信号の準備
let n = 1024;
let dt = 0.001;  // 1 kHz サンプリング

let signal: Array1<f64> = Array1::linspace(0.0, (n as f64) * dt, n)
    .mapv(|t| (2.0 * std::f64::consts::PI * 50.0 * t).sin());

// FFT実行
let spectrum = fft(&signal, None)?;

// 周波数軸の取得
let freqs = fftfreq(n, dt)?;

// 逆FFTで元の信号を復元
let reconstructed = ifft(&spectrum, None)?;
```

## 実数FFT

実数入力に特化したFFTで、計算量とメモリ使用量を半減できます。

```rust
use scirs2_fft::{rfft, irfft};

// 実数信号のFFT（出力はN/2+1個の複素数）
let spectrum = rfft(&signal, None)?;

// 逆変換
let reconstructed = irfft(&spectrum, Some(n))?;
```

## 周波数ユーティリティ

```rust
use scirs2_fft::{fftfreq, rfftfreq, fftshift};

// FFT周波数ビン
let freqs = fftfreq(1024, 0.001)?;       // 両側スペクトル
let rfreqs = rfftfreq(1024, 0.001)?;      // 片側スペクトル

// ゼロ周波数を中心にシフト
let shifted = fftshift(&spectrum)?;
```

## SciPy対応表

| SciPy | SciRS2 | 説明 |
|-------|--------|------|
| `scipy.fft.fft` | `scirs2_fft::fft` | 離散フーリエ変換 |
| `scipy.fft.ifft` | `scirs2_fft::ifft` | 逆離散フーリエ変換 |
| `scipy.fft.rfft` | `scirs2_fft::rfft` | 実数FFT |
| `scipy.fft.fftfreq` | `scirs2_fft::fftfreq` | 周波数ビン |
| `scipy.fft.fftshift` | `scirs2_fft::fftshift` | スペクトルシフト |
| `scipy.fft.dct` | `scirs2_fft::dct` | 離散コサイン変換 |
