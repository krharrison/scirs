# クイックスタート

このガイドでは、SciRS2の基本的な使い方を実際のコード例と共に紹介します。

## プロジェクトの作成

```bash
cargo new scirs2-example
cd scirs2-example
```

`Cargo.toml` に以下を追加します。

```toml
[dependencies]
scirs2-core = "0.4.0"
scirs2-linalg = "0.4.0"
scirs2-stats = "0.4.0"
scirs2-optimize = "0.4.0"
```

## 線形代数の基本

行列演算はSciRS2の最も基本的な操作です。

```rust
use scirs2_core::ndarray::{array, Array2};
use scirs2_linalg::{det, inv, solve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 連立一次方程式 Ax = b を解く
    let a = array![[3.0, 1.0], [1.0, 2.0]];
    let b = array![9.0, 8.0];

    let x = solve(&a.view(), &b.view())?;
    println!("解: {}", x);  // [2.0, 3.0]

    // 固有値分解
    // SciPyの scipy.linalg.eig に相当
    let matrix = array![[2.0, 1.0], [1.0, 3.0]];
    let eigenvalues = scirs2_linalg::eigh_vals(&matrix.view())?;
    println!("固有値: {}", eigenvalues);

    Ok(())
}
```

## 統計分析

確率分布の操作と統計検定を行います。

```rust
use scirs2_stats::distributions::{Normal, Distribution, ContinuousDistribution};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 正規分布 N(mu=0, sigma=1)
    let normal = Normal::new(0.0, 1.0)?;

    // 確率密度関数 (PDF)
    let pdf_val = normal.pdf(0.0);
    println!("PDF(0) = {:.4}", pdf_val);  // 0.3989

    // 累積分布関数 (CDF)
    let cdf_val = normal.cdf(1.96)?;
    println!("CDF(1.96) = {:.4}", cdf_val);  // 0.9750

    // パーセント点関数（逆CDF）
    let ppf_val = normal.ppf(0.975)?;
    println!("PPF(0.975) = {:.4}", ppf_val);  // 1.9600

    Ok(())
}
```

## 最適化

関数の最小化を行います。

```rust
use scirs2_optimize::minimize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rosenbrock関数の最小化
    // f(x, y) = (1-x)^2 + 100*(y-x^2)^2
    let rosenbrock = |x: &[f64]| -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    };

    // 勾配関数
    let grad = |x: &[f64]| -> Vec<f64> {
        let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        let dy = 200.0 * (x[1] - x[0] * x[0]);
        vec![dx, dy]
    };

    // 初期値
    let x0 = vec![0.0, 0.0];

    // L-BFGS法で最適化
    let result = minimize(rosenbrock, &x0, Some(grad), None)?;
    println!("最小点: {:?}", result.x);     // [1.0, 1.0] に近い値
    println!("最小値: {:.6}", result.fun);   // 0.0 に近い値

    Ok(())
}
```

## 信号処理

フィルタ設計と信号のフィルタリングを行います。

```rust
use scirs2_core::ndarray::Array1;
use scirs2_signal::filter_design::butter;
use scirs2_signal::filtering::filtfilt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // サンプリング周波数 1000 Hz の信号を生成
    let fs = 1000.0;
    let t: Array1<f64> = Array1::linspace(0.0, 1.0, 1000);

    // 50 Hz と 200 Hz の正弦波の合成
    let signal: Array1<f64> = t.mapv(|ti| {
        (2.0 * std::f64::consts::PI * 50.0 * ti).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 200.0 * ti).sin()
    });

    // 4次バターワースローパスフィルタ（カットオフ100 Hz）
    let (b, a) = butter(4, &[100.0 / (fs / 2.0)], None, None, None)?;

    // ゼロ位相フィルタリング
    let filtered = filtfilt(&b, &a, &signal)?;
    println!("フィルタリング完了: {} サンプル", filtered.len());

    Ok(())
}
```

## FFT（高速フーリエ変換）

周波数領域での信号解析を行います。

```rust
use scirs2_core::ndarray::Array1;
use scirs2_fft::{fft, fftfreq};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1024;
    let fs = 1000.0;
    let dt = 1.0 / fs;

    // テスト信号: 50 Hz + 120 Hz
    let t: Array1<f64> = Array1::linspace(0.0, (n as f64) * dt, n);
    let signal: Array1<f64> = t.mapv(|ti| {
        (2.0 * std::f64::consts::PI * 50.0 * ti).sin()
            + 0.8 * (2.0 * std::f64::consts::PI * 120.0 * ti).sin()
    });

    // FFTを実行
    let spectrum = fft(&signal, None)?;
    let freqs = fftfreq(n, dt)?;

    // パワースペクトルを計算
    let power: Array1<f64> = spectrum.mapv(|c| c.norm_sqr() / (n as f64));

    println!("FFT完了: {} 周波数ビン", freqs.len());

    Ok(())
}
```

## 次のステップ

- [プロジェクト構成](./structure.md) で全モジュールの概要を確認する
- [モジュール別ドキュメント](../modules/linalg.md) で詳細なAPIを学ぶ
- [SciPyからの移行ガイド](../migration/scipy.md) でPythonからの移行方法を確認する
